
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Fuses features from multiple encoders using Cross-Attention.
    Treats each encoder's output as a token in a sequence.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Multi-head attention layer
        # batch_first=True expects (Batch, Seq, Feature)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, feature_list):
        """
        Args:
            feature_list: List of tensors, each (B, hidden_dim)
        Returns:
            fused: (B, hidden_dim)
        """
        # Stack inputs to form a sequence: (B, num_encoders, hidden_dim)
        # e.g. [Short, Mid, Long, Context, Sentiment] -> 5 tokens
        x = torch.stack(feature_list, dim=1)
        
        # Self-Attention: Each feature source attends to every other source
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        
        # Residual + Norm
        x = self.norm1(x + attn_out)
        
        # FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Weighted Pooling (Average for now, could be learned attention pooling)
        # (B, num_encoders, hidden_dim) -> (B, hidden_dim)
        fused = x.mean(dim=1)
        
        return fused, attn_weights
