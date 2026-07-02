"""
Horizon-specialized models for the ensemble architecture.
Each model focuses on a specific time scale using the optimal encoder type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ResNLSEncoder, BiGRUEncoder, BiLSTMEncoder, ContextEncoder
from .sentiment_encoder import SentimentEncoder
from .stockxpert import ResidualBlock

class ThreeWayFusion(nn.Module):
    """
    Simplified fusion for 3 inputs: Temporal, Context, and Sentiment.
    Uses multi-head attention to mix these three diverse signal sources.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Deep interaction layer
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, temporal_enc, context_enc, sent_enc):
        """
        Args:
            temporal_enc: (B, hidden_dim) - The primary signal (ResNLS/GRU/LSTM)
            context_enc: (B, hidden_dim) - Market context
            sent_enc: (B, hidden_dim) - Sentiment signal
        Returns:
            global_fused: (B, hidden_dim)
        """
        # Stack tokens: (B, 3, hidden_dim)
        tokens = torch.stack([temporal_enc, context_enc, sent_enc], dim=1)
        
        # Self-attention
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)
        
        # Global pooling (mean of the three refined tokens)
        return tokens.mean(dim=1)

class specialized_head_block(nn.Module):
    """Helper to create prediction heads"""
    def __init__(self, input_dim, dropout, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(64, dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        return self.net(x)

class BaseHorizonModel(nn.Module):
    """Base class for horizon-specialized models."""
    def __init__(self, num_stocks, context_dim, hidden_dim, stock_embed_dim, attn_heads, dropout, output_horizons):
        super().__init__()
        self.output_horizons = output_horizons # List of horizon indices (e.g., [0] for H1, [1,2] for H3,H5)
        
        # Shared components
        self.context_encoder = ContextEncoder(context_dim, hidden_dim, dropout)
        self.sentiment_encoder = SentimentEncoder(9, hidden_dim, dropout)
        self.fusion = ThreeWayFusion(hidden_dim, attn_heads, dropout)
        self.stock_embedding = nn.Embedding(num_stocks, stock_embed_dim)
        
        self.combined_dim = hidden_dim + stock_embed_dim
        self.dropout_val = dropout
        
        # Heads - dynamically created based on output_horizons
        # We store them in a ModuleDict keyed by horizon index to allow sparse outputs
        self.direction_heads = nn.ModuleDict()
        self.magnitude_heads = nn.ModuleDict()
        self.level_heads = nn.ModuleDict()
        
        for h_idx in output_horizons:
            self.direction_heads[str(h_idx)] = specialized_head_block(self.combined_dim, dropout, 1)
            self.magnitude_heads[str(h_idx)] = specialized_head_block(self.combined_dim, dropout, 1)
            self.level_heads[str(h_idx)] = specialized_head_block(self.combined_dim, dropout, 3)
            
        # Aux tasks shared
        self.confidence_head = nn.Linear(self.combined_dim, len(output_horizons))
        self.target_zone_head = nn.Linear(self.combined_dim, len(output_horizons) * 7)

    def _forward_common(self, temporal_emb, context_x, sentiment_x, stock_idx):
        context_emb = self.context_encoder(context_x)
        sent_emb = self.sentiment_encoder(sentiment_x)
        
        # Fuse
        global_repr = self.fusion(temporal_emb, context_emb, sent_emb)
        
        # Add stock embedding
        stock_emb = self.stock_embedding(stock_idx)
        stock_emb = F.dropout(stock_emb, p=0.1, training=self.training)
        
        # Final combined representation
        combined = torch.cat([global_repr, stock_emb], dim=1)
        return combined

    def predict_heads(self, combined_repr):
        """Standard prediction logic for active horizons"""
        dir_preds = {}
        mag_preds = {}
        level_preds = {}
        
        for h_idx in self.output_horizons:
            k = str(h_idx)
            dir_preds[h_idx] = self.direction_heads[k](combined_repr)
            mag_preds[h_idx] = self.magnitude_heads[k](combined_repr)
            level_preds[h_idx] = self.level_heads[k](combined_repr)
            
        return dir_preds, mag_preds, level_preds

    def _pack_outputs(self, dir_preds, mag_preds, level_preds, confidence, tz_logits, batch_size, device):
        """Pad outputs to (B, 5, ...) for compatibility"""
        num_horizons = 5
        
        direction_logits = torch.zeros(batch_size, num_horizons, device=device)
        magnitude_pred = torch.zeros(batch_size, num_horizons, device=device)
        level_pred = torch.zeros(batch_size, num_horizons, 3, device=device)
        full_confidence = torch.zeros(batch_size, num_horizons, device=device)
        full_tz_logits = torch.zeros(batch_size, num_horizons, 7, device=device)
        
        for i, h_idx in enumerate(self.output_horizons):
            direction_logits[:, h_idx] = dir_preds[h_idx].squeeze(-1)
            magnitude_pred[:, h_idx] = mag_preds[h_idx].squeeze(-1)
            level_pred[:, h_idx] = level_preds[h_idx]
            full_confidence[:, h_idx] = confidence[:, i]
            full_tz_logits[:, h_idx] = tz_logits[:, i]
            
        return direction_logits, magnitude_pred, full_confidence, full_tz_logits, None, level_pred

class ShortTermModel(BaseHorizonModel):
    """
    Specialized for H1 (1-Day).
    Uses ResNLSEncoder for optimal short-term momentum capture.
    """
    def __init__(self, num_stocks, short_dim, context_dim, hidden_dim=96, stock_embed_dim=16, attn_heads=4, dropout=0.15, **kwargs):
        # Output H1 only (index 0)
        super().__init__(num_stocks, context_dim, hidden_dim, stock_embed_dim, attn_heads, dropout, output_horizons=[0])
        self.short_encoder = ResNLSEncoder(short_dim, hidden_dim, dropout)
        
    def forward(self, X_short, X_mid, X_long, X_context, X_sentiment, stock_idx):
        # Only use X_short
        temporal_emb = self.short_encoder(X_short)
        combined = self._forward_common(temporal_emb, X_context, X_sentiment, stock_idx)
        
        dir_preds, mag_preds, level_preds = self.predict_heads(combined)
        
        # Aux heads (only for 1 horizon)
        confidence = torch.sigmoid(self.confidence_head(combined))
        tz_logits = self.target_zone_head(combined).view(-1, 1, 7)
        
        return self._pack_outputs(dir_preds, mag_preds, level_preds, confidence, tz_logits, X_short.shape[0], X_short.device)

class MediumTermModel(BaseHorizonModel):
    """
    Specialized for H3, H5 (3-5 Days).
    Uses BiGRUEncoder for efficient mid-range trend modeling.
    """
    def __init__(self, num_stocks, mid_dim, context_dim, hidden_dim=96, stock_embed_dim=16, attn_heads=4, dropout=0.15, **kwargs):
        # Output H3 (idx 1) and H5 (idx 2)
        super().__init__(num_stocks, context_dim, hidden_dim, stock_embed_dim, attn_heads, dropout, output_horizons=[1, 2])
        self.mid_encoder = BiGRUEncoder(mid_dim, hidden_dim, num_layers=3, dropout=dropout)
        
    def forward(self, X_short, X_mid, X_long, X_context, X_sentiment, stock_idx):
        # Only use X_mid
        temporal_emb = self.mid_encoder(X_mid)
        combined = self._forward_common(temporal_emb, X_context, X_sentiment, stock_idx)
        
        dir_preds, mag_preds, level_preds = self.predict_heads(combined)
        
        # Aux heads
        confidence = torch.sigmoid(self.confidence_head(combined))
        tz_logits = self.target_zone_head(combined).view(-1, 2, 7)
        
        return self._pack_outputs(dir_preds, mag_preds, level_preds, confidence, tz_logits, X_mid.shape[0], X_mid.device)

class LongTermModel(BaseHorizonModel):
    """
    Specialized for H7, H10 (7-10 Days).
    Uses BiLSTMEncoder with Time2Vec for capturing long-term dependencies and cycles.
    """
    def __init__(self, num_stocks, long_dim, context_dim, hidden_dim=96, stock_embed_dim=16, attn_heads=4, dropout=0.15, **kwargs):
        # Output H7 (idx 3) and H10 (idx 4)
        super().__init__(num_stocks, context_dim, hidden_dim, stock_embed_dim, attn_heads, dropout, output_horizons=[3, 4])
        self.long_encoder = BiLSTMEncoder(long_dim, hidden_dim, num_layers=3, dropout=dropout)
        
    def forward(self, X_short, X_mid, X_long, X_context, X_sentiment, stock_idx):
        # Only use X_long
        temporal_emb = self.long_encoder(X_long)
        combined = self._forward_common(temporal_emb, X_context, X_sentiment, stock_idx)
        
        dir_preds, mag_preds, level_preds = self.predict_heads(combined)
        
        confidence = torch.sigmoid(self.confidence_head(combined))
        tz_logits = self.target_zone_head(combined).view(-1, 2, 7)
        
        return self._pack_outputs(dir_preds, mag_preds, level_preds, confidence, tz_logits, X_long.shape[0], X_long.device)
