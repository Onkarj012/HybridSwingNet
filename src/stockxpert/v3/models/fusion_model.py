"""V3 multi-encoder fusion model variant."""

from __future__ import annotations

import torch
import torch.nn as nn

from stockxpert.models.context_model import ContextEncoder
from stockxpert.models.encoders import BiGRUEncoder, BiLSTMEncoder, ResNLSEncoder
from stockxpert.models.fusion_layers import CrossAttentionFusion


class MultiEncoderFusionModel(nn.Module):
    """Fusion model using static stock traits instead of learned stock IDs."""

    def __init__(
        self,
        short_dim: int,
        mid_dim: int,
        long_dim: int,
        context_dim: int,
        sentiment_dim: int = 1,
        hidden_dim: int = 96,
        stock_embed_dim: int = 16,
        stock_traits_dim: int = 7,
        attn_heads: int = 4,
        dropout: float = 0.15,
        output_horizons: list[int] | None = None,
    ):
        super().__init__()
        self.output_horizons = output_horizons or [0, 1, 2, 3, 4]
        self.short_encoder = ResNLSEncoder(short_dim, hidden_dim, dropout)
        self.mid_encoder = BiGRUEncoder(mid_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.long_encoder = BiLSTMEncoder(long_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.context_encoder = ContextEncoder(context_dim, hidden_dim, dropout)
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stock_traits_proj = nn.Sequential(nn.Linear(stock_traits_dim, stock_embed_dim), nn.GELU(), nn.Linear(stock_embed_dim, hidden_dim))
        self.fusion = CrossAttentionFusion(hidden_dim, attn_heads, dropout)

        self.magnitude_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)) for _ in range(5)])
        self.direction_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)) for _ in range(5)])
        self.zone_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 7)) for _ in range(5)])
        self.level_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 3)) for _ in range(5)])
        self.confidence_head = nn.Linear(hidden_dim, 5)
        self.aux_trend_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_short, x_mid, x_long, x_context, x_sentiment, stock_traits):
        features = [
            self.short_encoder(x_short),
            self.mid_encoder(x_mid),
            self.long_encoder(x_long),
            self.context_encoder(x_context),
            self.sentiment_encoder(x_sentiment),
            self.stock_traits_proj(stock_traits.float()),
        ]
        h_fused = self.fusion(features)
        magnitude_pred = torch.cat([head(h_fused) for head in self.magnitude_heads], dim=1)
        direction_logits = torch.cat([head(h_fused) for head in self.direction_heads], dim=1)
        target_zone_logits = torch.cat([head(h_fused).unsqueeze(1) for head in self.zone_heads], dim=1)
        level_pred = torch.cat([head(h_fused).unsqueeze(1) for head in self.level_heads], dim=1)
        confidence = torch.sigmoid(self.confidence_head(h_fused))
        aux_trend_logit = self.aux_trend_head(h_fused)
        return direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred

