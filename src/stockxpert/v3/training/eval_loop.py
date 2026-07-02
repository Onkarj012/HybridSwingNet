"""V3 evaluation loop."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    horizons = horizons or [1, 3, 5, 7, 10]
    model.eval()

    all_dir_preds = []
    all_dir_targets = []
    all_mag_preds = []
    all_mag_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            stock_traits = inputs["stock_traits"].to(device)
            outputs = model(
                inputs["X_short"].to(device),
                inputs["X_mid"].to(device),
                inputs["X_long"].to(device),
                inputs["X_context"].to(device),
                inputs["X_sentiment"].to(device),
                stock_traits,
            )
            direction_logits = outputs[0]
            magnitude_pred = outputs[1]
            mag_targets = targets["magnitude"].to(device) if isinstance(targets, dict) else targets.to(device)

            dir_probs = torch.sigmoid(direction_logits)
            all_dir_preds.append((dir_probs > 0.5).int().cpu().numpy())
            all_dir_targets.append((mag_targets > 0).int().cpu().numpy())
            all_mag_preds.append(magnitude_pred.cpu().numpy())
            all_mag_targets.append(mag_targets.cpu().numpy())

    all_dir_preds = np.concatenate(all_dir_preds, axis=0)
    all_dir_targets = np.concatenate(all_dir_targets, axis=0)
    all_mag_preds = np.concatenate(all_mag_preds, axis=0)
    all_mag_targets = np.concatenate(all_mag_targets, axis=0)

    metrics_per_horizon = []
    for i, h in enumerate(horizons):
        correct = all_dir_preds[:, i] == all_dir_targets[:, i]
        metrics_per_horizon.append(
            {
                "horizon": h,
                "direction_accuracy": float(correct.mean()),
                "mae": float(np.mean(np.abs(all_mag_preds[:, i] - all_mag_targets[:, i]))),
                "rmse": float(np.sqrt(np.mean((all_mag_preds[:, i] - all_mag_targets[:, i]) ** 2))),
            }
        )

    return {"metrics_per_horizon": metrics_per_horizon}
