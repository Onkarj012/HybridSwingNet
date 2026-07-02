"""V3 loss functions."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from stockxpert.models.losses import DirectionAwareLoss, StockXpertLoss as LegacyStockXpertLoss


class StockXpertV3Loss(nn.Module):
    """Trimmed V3 objective with optional legacy-loss fallback."""

    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.4,
        regime_weight: float = 0.2,
        confidence_weight: float = 0.0,
        aux_trend_weight: float = 0.0,
        consistency_weight: float = 0.0,
        legacy_loss_mode: bool = False,
        horizon_weights: Optional[list[float]] = None,
        gamma_pos: float = 1.5,
        gamma_neg: float = 1.5,
        downside_weight: float = 1.0,
        **legacy_kwargs,
    ):
        super().__init__()
        self.legacy_loss_mode = legacy_loss_mode
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.regime_weight = regime_weight
        self.confidence_weight = confidence_weight
        self.aux_trend_weight = aux_trend_weight
        self.consistency_weight = consistency_weight
        self.direction_loss = DirectionAwareLoss(gamma_pos, gamma_neg, downside_weight)
        self.magnitude_loss = nn.HuberLoss(reduction="none")
        self.regime_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.BCELoss(reduction="none")
        self.aux_trend_loss = nn.BCEWithLogitsLoss()
        self.horizon_weights = torch.tensor(horizon_weights, dtype=torch.float32) if horizon_weights else None

        self.legacy_loss = None
        if legacy_loss_mode:
            self.legacy_loss = LegacyStockXpertLoss(
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
                horizon_weights=horizon_weights,
                **legacy_kwargs,
            )

    def forward(
        self,
        direction_logits: torch.Tensor,
        magnitude_pred: torch.Tensor,
        confidence: torch.Tensor,
        target_zone_logits: torch.Tensor,
        aux_trend_logit: Optional[torch.Tensor],
        level_pred: Optional[torch.Tensor],
        targets: Dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.legacy_loss_mode:
            if self.legacy_loss is None:
                raise RuntimeError("legacy_loss_mode=True but legacy loss was not initialized")
            return self.legacy_loss(
                direction_logits,
                magnitude_pred,
                confidence,
                target_zone_logits,
                aux_trend_logit,
                level_pred,
                targets,
            )

        mag_targets = targets["magnitude"] if isinstance(targets, dict) else targets
        direction_targets = (mag_targets > 0).float()

        weights = None
        if self.horizon_weights is not None:
            weights = self.horizon_weights.to(direction_logits.device).view(1, -1)

        loss_direction = self.direction_loss(direction_logits, direction_targets, weights=weights)
        loss_magnitude_raw = self.magnitude_loss(magnitude_pred, mag_targets)
        loss_magnitude = (loss_magnitude_raw * weights).mean() if weights is not None else loss_magnitude_raw.mean()

        loss_regime = torch.tensor(0.0, device=direction_logits.device)
        if isinstance(targets, dict) and "regime" in targets and "regime_logits" in targets:
            loss_regime = self.regime_loss(targets["regime_logits"], targets["regime"].long())

        loss_confidence = torch.tensor(0.0, device=direction_logits.device)
        if self.confidence_weight > 0:
            probs = torch.sigmoid(direction_logits).detach()
            direction_correct_prob = torch.where(direction_targets > 0.5, probs, 1.0 - probs)
            confidence_target = direction_correct_prob.clamp(0.0, 1.0)
            confidence_raw = self.confidence_loss(confidence.clamp(1e-6, 1.0 - 1e-6), confidence_target)
            loss_confidence = (confidence_raw * weights).mean() if weights is not None else confidence_raw.mean()

        loss_aux_trend = torch.tensor(0.0, device=direction_logits.device)
        if self.aux_trend_weight > 0 and aux_trend_logit is not None:
            if isinstance(targets, dict) and "trend" in targets:
                trend_target = targets["trend"].float().view_as(aux_trend_logit)
            else:
                trend_target = (mag_targets.mean(dim=1, keepdim=True) > 0).float()
            loss_aux_trend = self.aux_trend_loss(aux_trend_logit, trend_target)

        loss_consistency = torch.tensor(0.0, device=direction_logits.device)
        if self.consistency_weight > 0 and aux_trend_logit is not None:
            horizon_direction = torch.sigmoid(direction_logits).mean(dim=1, keepdim=True)
            aux_direction = torch.sigmoid(aux_trend_logit)
            loss_consistency = torch.mean(torch.abs(horizon_direction - aux_direction))

        total_loss = (
            self.direction_weight * loss_direction
            + self.magnitude_weight * loss_magnitude
            + self.regime_weight * loss_regime
            + self.confidence_weight * loss_confidence
            + self.aux_trend_weight * loss_aux_trend
            + self.consistency_weight * loss_consistency
        )

        return total_loss, {
            "total": float(total_loss.detach().cpu()),
            "direction": float(loss_direction.detach().cpu()),
            "magnitude": float(loss_magnitude.detach().cpu()),
            "regime": float(loss_regime.detach().cpu()),
            "confidence": float(loss_confidence.detach().cpu()),
            "aux_trend": float(loss_aux_trend.detach().cpu()),
            "consistency": float(loss_consistency.detach().cpu()),
        }


StockXpertLoss = StockXpertV3Loss
