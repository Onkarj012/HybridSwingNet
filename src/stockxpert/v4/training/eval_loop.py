"""V4 evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def predict_frame(model, dataloader, device, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    sample_offset = 0
    dataset = dataloader.dataset
    with torch.no_grad():
        for inputs, targets in dataloader:
            batch_size = inputs["X_short"].shape[0]
            outputs = model(
                inputs["X_short"].to(device),
                inputs["X_mid"].to(device),
                inputs["X_long"].to(device),
                inputs["X_context"].to(device),
                inputs["X_sentiment"].to(device),
                inputs["stock_traits"].to(device),
            )
            p_up = torch.sigmoid(outputs["p_up_logits"]).cpu().numpy()
            action = outputs["action_logits"].argmax(dim=-1).cpu().numpy()
            long_ret = outputs["expected_long_return"].cpu().numpy()
            short_ret = outputs["expected_short_return"].cpu().numpy()
            risk = outputs["downside_risk"].cpu().numpy()
            for i in range(batch_size):
                sample = dataset.samples[sample_offset + i]
                for h_idx, horizon in enumerate(horizons):
                    realized = sample.net_return_long if action[i, h_idx] != 2 else sample.net_return_short
                    rows.append(
                        {
                            "date": pd.to_datetime(sample.date).strftime("%Y-%m-%d"),
                            "symbol": sample.symbol,
                            "horizon": horizon,
                            "p_up": float(p_up[i, h_idx]),
                            "predicted_action": int(action[i, h_idx]),
                            "actual_action": int(sample.trade_action),
                            "expected_long_return": float(long_ret[i, h_idx]),
                            "expected_short_return": float(short_ret[i, h_idx]),
                            "downside_risk": float(risk[i, h_idx]),
                            "cost_bps": float(sample.cost_bps),
                            "realized_long_return": float(sample.net_return_long),
                            "realized_short_return": float(sample.net_return_short),
                            "realized_net_return": float(realized),
                        }
                    )
            sample_offset += batch_size
    return pd.DataFrame(rows)


def evaluate_v4(model, dataloader, device, horizons: list[int]) -> dict[str, Any]:
    preds = predict_frame(model, dataloader, device, horizons)
    if preds.empty:
        return {"metrics_per_horizon": []}
    metrics = []
    for horizon, group in preds.groupby("horizon"):
        p_up_pred = (group["p_up"] >= 0.5).to_numpy()
        p_up_label = (group["realized_long_return"] >= group["realized_short_return"]).to_numpy()
        action_acc = (group["predicted_action"].to_numpy() == group["actual_action"].to_numpy()).mean()
        metrics.append(
            {
                "horizon": int(horizon),
                "p_up_accuracy": float(np.mean(p_up_pred == p_up_label)),
                "action_accuracy": float(action_acc),
                "avg_expected_long_return": float(group["expected_long_return"].mean()),
                "avg_realized_long_return": float(group["realized_long_return"].mean()),
            }
        )
    return {"metrics_per_horizon": metrics}
