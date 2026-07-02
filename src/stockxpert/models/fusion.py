"""
Multi-head attention fusion for combining encoded tokens.
"""

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """
    Fuse multiple encoder outputs using multi-head attention and horizon-aware pooling.
    
    Input: encoder tokens (short, mid, long, context, sentiment)
    Output: tuple(global fused representation, horizon-specific fused representations)
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.15,
        num_horizons: int = 5,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_horizons = num_horizons
        self.scale = hidden_dim ** 0.5
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Position-wise FeedForward (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        
        # Learnable horizon embeddings act as horizon-specific queries
        self.horizon_embeddings = nn.Parameter(
            torch.randn(num_horizons, hidden_dim) * 0.02
        )
        
        # Second Attention Layer (Temporal Fusion Transformer style)
        self.attention2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Cross Attention Layer (Horizon queries -> Encoder keys)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, short_enc, mid_enc, long_enc, context_enc, sent_enc):
        """
        Args:
            short_enc: (B, hidden_dim)
            mid_enc: (B, hidden_dim)
            long_enc: (B, hidden_dim)
            context_enc: (B, hidden_dim)
            sent_enc: (B, hidden_dim)
        
        Returns:
            global_fused: (B, hidden_dim)
            fused_per_horizon: (B, num_horizons, hidden_dim)
        """
        # Stack tokens: (B, 5, hidden_dim)
        tokens = torch.stack([short_enc, mid_enc, long_enc, context_enc, sent_enc], dim=1)
        
        # 1. First Self-attention Block
        # Capture weights: (B, L, S) where L=S=5
        attn_out, attn_weights_1 = self.attention(tokens, tokens, tokens, need_weights=True)
        tokens = self.layer_norm(tokens + self.dropout(attn_out))
        
        # 2. FFN Block
        ffn_out = self.ffn(tokens)
        tokens = self.ln_ffn(tokens + ffn_out)

        # 3. Second Self-attention Block (Deep Interaction)
        attn_out2, attn_weights_2 = self.attention2(tokens, tokens, tokens, need_weights=True)
        tokens_deep = self.layer_norm2(tokens + self.dropout(attn_out2))
        
        # Gated Residual Connection (GRC)
        gate_input = torch.cat([tokens, tokens_deep], dim=-1)
        gate_val = torch.sigmoid(self.gate(gate_input))
        tokens = gate_val * tokens_deep + (1 - gate_val) * tokens
        
        # 4. Cross-Attention: Horizon Embeddings attend to Encoder Tokens
        # Query: Horizon Embeddings (H, hidden_dim) -> expanded to (B, H, hidden_dim)
        # Key/Value: Encoder Tokens (B, 5, hidden_dim)
        B = tokens.size(0)
        
        # Expand horizon embeddings to batch
        queries = self.horizon_embeddings.unsqueeze(0).expand(B, -1, -1) # (B, H, hidden)
        
        # Cross-attention (using built-in MultiheadAttention)
        # Returns (output, weights) 
        # weights shape: (B, L_query, S_key) -> (B, num_horizons, 5)
        cross_out, cross_attn_weights = self.cross_attention(
            queries,     # Query
            tokens,      # Key
            tokens,      # Value
            need_weights=True
        )
        fused_per_horizon = cross_out # (B, H, hidden)
        
        # Global fused via mean pooling
        global_fused = tokens.mean(dim=1)
        
        attention_dict = {
            "self_attn_1": attn_weights_1, # (B, 5, 5)
            "self_attn_2": attn_weights_2, # (B, 5, 5)
            "cross_attn": cross_attn_weights # (B, num_horizons, 5)
        }
        
        return global_fused, fused_per_horizon, attention_dict


class HorizonMixer(nn.Module):
    """
    Dedicated mixer for horizon-specific representations.
    Allows long horizons (H7, H10) to attend to H1/H3 signals.
    """
    def __init__(self, hidden_dim: int, num_horizons: int = 5, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
        # Dedicated trend predictor for H10 (Auxiliary Head)
        # Predicts 10-day SMA direction as a "hint"
        self.aux_trend_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, fused_per_horizon):
        """
        Args:
            fused_per_horizon: (B, num_horizons, hidden_dim)
        Returns:
            enhanced_horizons: (B, num_horizons, hidden_dim)
            aux_trend_logit: (B, 1) or None
        """
        # Create causal mask for self-attention
        # Shape: (num_horizons, num_horizons)
        # H1 (idx 0) can attend to H1 only
        # H10 (idx 4) can attend to H1, H3, H5, H7, H10
        B, H, D = fused_per_horizon.shape
        device = fused_per_horizon.device
        
        # Standard causal mask (triangular upper)
        # Positions with float('-inf') are masked out
        mask = torch.triu(torch.ones(H, H, device=device) * float('-inf'), diagonal=1)
        
        # Mix information across horizons with causal constraint
        enhanced_horizons = self.transformer(fused_per_horizon, mask=mask)
        
        # For H10 (index 4), predict trend
        # We assume H10 is the last horizon
        h10_repr = enhanced_horizons[:, -1, :]
        aux_trend_logit = self.aux_trend_head(h10_repr)
        
        return enhanced_horizons, aux_trend_logit
