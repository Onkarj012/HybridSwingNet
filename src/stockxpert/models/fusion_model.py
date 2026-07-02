
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from stockxpert.models.encoders import ResNLSEncoder, BiGRUEncoder, BiLSTMEncoder
from stockxpert.models.context_model import ContextEncoder
from stockxpert.models.fusion_layers import CrossAttentionFusion

class MultiEncoderFusionModel(nn.Module):
    """
    Unified Ensemble Architecture.
    
    1. Extracts features using SPECIALIZED encoders (Short, Mid, Long).
    2. Adds Context and Sentiment representations.
    3. FUSES all features using Cross-Attention.
    4. Predicts ALL horizons simultaneously from the fused state.
    """
    def __init__(
        self,
        num_stocks: int,
        short_dim: int,
        mid_dim: int,
        long_dim: int,
        context_dim: int,
        sentiment_dim: int = 1,
        hidden_dim: int = 96,
        stock_embed_dim: int = 16,
        attn_heads: int = 4,
        dropout: float = 0.15,
        output_horizons: List[int] = [0, 1, 2, 3, 4]  # Indices for 1,3,5,7,10
    ):
        super().__init__()
        
        self.output_horizons = output_horizons
        
        # 1. Specialized Encoders
        self.short_encoder = ResNLSEncoder(short_dim, hidden_dim, dropout)
        self.mid_encoder = BiGRUEncoder(mid_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.long_encoder = BiLSTMEncoder(long_dim, hidden_dim, num_layers=2, dropout=dropout)
        
        # 2. Auxiliary Encoders
        self.context_encoder = ContextEncoder(context_dim, hidden_dim, dropout)
        
        # Sentiment Encoder
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stock Embedding (Static Identity)
        self.stock_embedding = nn.Embedding(num_stocks, stock_embed_dim)
        self.stock_proj = nn.Linear(stock_embed_dim, hidden_dim)

        # 3. Fusion Layer
        self.fusion = CrossAttentionFusion(hidden_dim, attn_heads, dropout)
        
        # 4. Multi-Horizon Prediction Heads
        
        # Magnitude Heads (Regression)
        self.magnitude_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(5)
        ])
        
        # Direction Heads (Logits)
        self.direction_heads = nn.ModuleList([
             nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(), # Keeping consistent activation
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(5)
        ])
        
        # Target Zone Heads (Classification - 7 classes)
        self.zone_heads = nn.ModuleList([
             nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 7)
            ) for _ in range(5)
        ])
        
        # Support/Resistance Level Heads (Regression - 3 vals: Res, Sup, Tgt)
        self.level_heads = nn.ModuleList([
             nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 3)
            ) for _ in range(5)
        ])
        
        # Shared Auxiliary Heads
        # Confidence: one per horizon, from global representation
        self.confidence_head = nn.Linear(hidden_dim, 5) 
        
        # Aux Trend: global trend context (binary logit)
        self.aux_trend_head = nn.Linear(hidden_dim, 1)
            
    def forward(self, x_short, x_mid, x_long, x_context, x_sentiment, stock_idx):
        """
        Args:
            x_short: (B, T_s, D_s)
            x_mid:   (B, T_m, D_m)
            x_long:  (B, T_l, D_l)
            x_context: (B, D_c)
            x_sentiment: (B, 1)
            stock_idx: (B,)
        
        Returns tuple compatible with train_loop:
           (direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred)
        """
        # 1. Feature Extraction
        h_short = self.short_encoder(x_short)     # (B, H)
        h_mid = self.mid_encoder(x_mid)           # (B, H)
        h_long = self.long_encoder(x_long)        # (B, H)
        h_ctx = self.context_encoder(x_context)   # (B, H)
        h_sent = self.sentiment_encoder(x_sentiment) # (B, H)
        
        # Stock Identity
        h_id = self.stock_embedding(stock_idx)
        h_id = self.stock_proj(h_id) # (B, H)
        
        # 2. Fusion
        # We fuse ALL feature sources including stock ID
        features = [h_short, h_mid, h_long, h_ctx, h_sent, h_id]
        
        h_fused = self.fusion(features) # (B, H)
        
        # 3. Prediction
        
        # Stack predictions for all 5 horizons
        mag_list = [head(h_fused) for head in self.magnitude_heads]
        magnitude_pred = torch.cat(mag_list, dim=1) # (B, 5)
        
        dir_list = [head(h_fused) for head in self.direction_heads]
        direction_logits = torch.cat(dir_list, dim=1) # (B, 5)
        
        zone_list = [head(h_fused).unsqueeze(1) for head in self.zone_heads] 
        target_zone_logits = torch.cat(zone_list, dim=1) # (B, 5, 7)
        
        level_list = [head(h_fused).unsqueeze(1) for head in self.level_heads]
        level_pred = torch.cat(level_list, dim=1) # (B, 5, 3)
        
        # Shared
        confidence = torch.sigmoid(self.confidence_head(h_fused)) # (B, 5)
        aux_trend_logit = self.aux_trend_head(h_fused) # (B, 1)
        
        return direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred
