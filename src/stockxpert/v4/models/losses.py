"""V4 multi-head trading losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockXpertV4Loss(nn.Module):
    def __init__(
        self,
        p_up_weight: float = 0.5,
        action_weight: float = 1.0,
        long_return_weight: float = 0.6,
        short_return_weight: float = 0.6,
        downside_weight: float = 0.4,
        confidence_weight: float = 0.2,
        aux_trend_weight: float = 0.1,
    ):
        super().__init__()
        self.weights = {
            "p_up": p_up_weight,
            "action": action_weight,
            "long_return": long_return_weight,
            "short_return": short_return_weight,
            "downside": downside_weight,
            "confidence": confidence_weight,
            "aux_trend": aux_trend_weight,
        }

    def forward(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        sample_weight = targets.get("sample_weight", torch.ones_like(targets["p_up"])).view(-1, 1)
        p_up_target = targets["p_up"].float().view(-1, 1).expand_as(outputs["p_up_logits"])
        action_target = targets["action"].long().view(-1, 1).expand(outputs["action_logits"].shape[:2])

        p_up = F.binary_cross_entropy_with_logits(outputs["p_up_logits"], p_up_target, reduction="none")
        action = F.cross_entropy(
            outputs["action_logits"].reshape(-1, 3),
            action_target.reshape(-1),
            reduction="none",
        ).view_as(action_target)
        long_ret = F.huber_loss(
            outputs["expected_long_return"],
            targets["net_return_long"].float().view(-1, 1).expand_as(outputs["expected_long_return"]),
            reduction="none",
        )
        short_ret = F.huber_loss(
            outputs["expected_short_return"],
            targets["net_return_short"].float().view(-1, 1).expand_as(outputs["expected_short_return"]),
            reduction="none",
        )
        downside = F.huber_loss(
            outputs["downside_risk"],
            targets["downside"].float().view(-1, 1).expand_as(outputs["downside_risk"]),
            reduction="none",
        )
        confidence = F.binary_cross_entropy(
            outputs["confidence_correctness"].clamp(1e-6, 1.0 - 1e-6),
            targets["confidence_correct"].float().view(-1, 1).expand_as(outputs["confidence_correctness"]),
            reduction="none",
        )

        aux = torch.tensor(0.0, device=sample_weight.device)
        if outputs.get("aux_trend_logit") is not None and "trend" in targets:
            aux = F.binary_cross_entropy_with_logits(outputs["aux_trend_logit"], targets["trend"].float())

        weighted = {
            "p_up": (p_up * sample_weight).mean(),
            "action": (action * sample_weight).mean(),
            "long_return": (long_ret * sample_weight).mean(),
            "short_return": (short_ret * sample_weight).mean(),
            "downside": (downside * sample_weight).mean(),
            "confidence": (confidence * sample_weight).mean(),
            "aux_trend": aux,
        }
        total = sum(self.weights[name] * value for name, value in weighted.items())
        metrics = {name: float(value.detach().cpu()) for name, value in weighted.items()}
        metrics["total"] = float(total.detach().cpu())
        return total, metrics


StockXpertLoss = StockXpertV4Loss
