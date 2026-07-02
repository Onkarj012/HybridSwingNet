"""V3 StockXpert model with static stock traits instead of stock ID embeddings."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stockxpert.models.encoders import BiGRUEncoder, BiLSTMEncoder, ContextEncoder, ResNLSEncoder
from stockxpert.models.fusion import AttentionFusion, HorizonMixer
from stockxpert.models.sentiment_encoder import SentimentEncoder
from stockxpert.models.stockxpert import RegimeDetector, ResidualBlock


class StockXpertModelV3(nn.Module):
    """HybridSwingNet V3 core model.

    V3 preserves the five-encoder attention architecture but removes learned
    stock identity embeddings. A static trait vector is projected into the same
    representation width previously used by the embedding.
    """

    def __init__(
        self,
        short_dim: int,
        mid_dim: int,
        long_dim: int,
        context_dim: int,
        num_horizons: int = 5,
        hidden_dim: int = 96,
        stock_embed_dim: int = 16,
        stock_traits_dim: int = 7,
        sentiment_dim: int = 9,
        attn_heads: int = 4,
        dropout: float = 0.15,
        bottleneck_dim: int = 64,
        use_regime_head: bool = False,
    ):
        super().__init__()
        self.num_horizons = num_horizons
        self.use_regime_head = use_regime_head

        self.short_encoder = ResNLSEncoder(short_dim, hidden_dim, dropout)
        self.mid_encoder = BiGRUEncoder(mid_dim, hidden_dim, dropout=dropout)
        self.long_encoder = BiLSTMEncoder(long_dim, hidden_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(context_dim, hidden_dim, dropout)
        self.sentiment_encoder = SentimentEncoder(sentiment_dim, hidden_dim, dropout)

        self.fusion = AttentionFusion(hidden_dim, attn_heads, dropout, num_horizons)
        self.mixer = HorizonMixer(hidden_dim, num_horizons, n_layers=2, dropout=dropout)

        self.stock_traits_proj = nn.Sequential(
            nn.Linear(stock_traits_dim, stock_embed_dim),
            nn.LayerNorm(stock_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        combined_dim = hidden_dim + stock_embed_dim

        def make_bottleneck_head(out_features: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(combined_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                ResidualBlock(bottleneck_dim, dropout),
                nn.Linear(bottleneck_dim, 32),
                nn.ReLU(),
                nn.Linear(32, out_features),
            )

        self.direction_heads = nn.ModuleList([make_bottleneck_head() for _ in range(num_horizons)])
        self.magnitude_heads = nn.ModuleList([make_bottleneck_head() for _ in range(num_horizons)])

        # Kept for output compatibility. V3 loss ignores confidence/zone/level by default.
        self.level_heads = nn.ModuleList([make_bottleneck_head(3) for _ in range(num_horizons)])
        self.confidence_heads = nn.ModuleList([nn.Linear(combined_dim, 1) for _ in range(num_horizons)])
        self.target_zone_heads = nn.ModuleList([nn.Linear(combined_dim, 7) for _ in range(num_horizons)])

        if use_regime_head:
            self.regime_detector = RegimeDetector(hidden_dim, dropout)

    def forward(
        self,
        X_short: torch.Tensor,
        X_mid: torch.Tensor,
        X_long: torch.Tensor,
        X_context: torch.Tensor,
        X_sentiment: torch.Tensor,
        stock_traits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict]:
        short_enc = self.short_encoder(X_short)
        mid_enc = self.mid_encoder(X_mid)
        long_enc = self.long_encoder(X_long)
        context_enc = self.context_encoder(X_context)
        sent_enc = self.sentiment_encoder(X_sentiment)

        global_fused, fused_per_horizon, attention_dict = self.fusion(
            short_enc, mid_enc, long_enc, context_enc, sent_enc
        )
        fused_per_horizon, aux_trend_logit = self.mixer(fused_per_horizon)

        stock_emb = self.stock_traits_proj(stock_traits.float())

        regime_logits = None
        if self.use_regime_head:
            regime_logits, regime_probs = self.regime_detector(global_fused)
            attention_dict = {**attention_dict, "regime_probs": regime_probs}

        dir_logits_list = []
        mag_pred_list = []
        level_pred_list = []
        conf_list = []
        tz_list = []

        for i in range(self.num_horizons):
            horizon_rep = fused_per_horizon[:, i, :]
            if self.num_horizons == 5:
                blend_weights = [
                    (0.7, 0.3, 0.0),
                    (0.3, 0.7, 0.0),
                    (0.0, 0.5, 0.5),
                    (0.0, 0.3, 0.7),
                    (0.0, 0.0, 1.0),
                ]
                w_s, w_m, w_l = blend_weights[i]
                horizon_rep = horizon_rep + w_s * short_enc + w_m * mid_enc + w_l * long_enc

            ctx = torch.cat([horizon_rep, stock_emb], dim=1)
            dir_logits_list.append(self.direction_heads[i](ctx))
            mag_pred_list.append(self.magnitude_heads[i](ctx))
            level_pred_list.append(self.level_heads[i](ctx))
            conf_list.append(self.confidence_heads[i](ctx))
            tz_list.append(self.target_zone_heads[i](ctx))

        direction_logits = torch.cat(dir_logits_list, dim=1)
        magnitude_pred = torch.tanh(torch.cat(mag_pred_list, dim=1)) * 5.0
        confidence = torch.sigmoid(torch.cat(conf_list, dim=1))
        target_zone_logits = torch.stack(tz_list, dim=1)
        level_pred = torch.stack(level_pred_list, dim=1)

        if regime_logits is not None:
            attention_dict = {**attention_dict, "regime_logits": regime_logits}

        return direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, attention_dict


StockXpertModel = StockXpertModelV3

