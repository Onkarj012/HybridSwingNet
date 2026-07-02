"""Nested walk-forward policy selection and replay."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from stockxpert.v4.policy.selection import PolicySettings, select_trades


def evaluate_trades(trades: pd.DataFrame, return_column: str = "realized_net_return") -> dict[str, float]:
    if trades.empty:
        return {"return_pct": 0.0, "profit_factor": 0.0, "max_dd_pct": 0.0, "trades": 0, "long_trades": 0, "short_trades": 0}
    if "date" in trades.columns:
        returns = trades.groupby("date")[return_column].mean().fillna(0.0).astype(float).to_numpy()
    else:
        returns = trades[return_column].fillna(0.0).astype(float).to_numpy()
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity / peaks) - 1.0
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return {
        "return_pct": float((equity[-1] - 1.0) * 100.0),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
        "max_dd_pct": float(abs(drawdowns.min()) * 100.0),
        "trades": int(len(trades)),
        "long_trades": int((trades["side"] == "LONG").sum()) if "side" in trades.columns else 0,
        "short_trades": int((trades["side"] == "SHORT").sum()) if "side" in trades.columns else 0,
    }


def reject_unstable_winner(trades: pd.DataFrame) -> bool:
    if trades.empty or "date" not in trades.columns or "realized_net_return" not in trades.columns:
        return True
    tmp = trades.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.to_period("M").astype(str)
    by_month = tmp.groupby("month")["realized_net_return"].sum()
    if len(by_month) > 1 and by_month.max() > max(0.0, tmp["realized_net_return"].sum()) * 0.8:
        return True
    if "side" in tmp.columns and tmp["side"].nunique() == 1 and len(tmp) >= 10:
        return True
    return False


def choose_policy(train_predictions: pd.DataFrame, candidates: Iterable[PolicySettings]) -> tuple[PolicySettings, pd.DataFrame]:
    rows = []
    best_settings = None
    best_score = -float("inf")
    for settings in candidates:
        trades = select_trades(train_predictions, settings)
        metrics = evaluate_trades(trades)
        rejected = reject_unstable_winner(trades)
        score = metrics["return_pct"] + 10.0 * metrics["profit_factor"] - 0.5 * metrics["max_dd_pct"]
        if rejected:
            score = -float("inf")
        rows.append({**asdict(settings), **metrics, "rejected": rejected, "score": score})
        if score > best_score:
            best_score = score
            best_settings = settings
    if best_settings is None:
        best_settings = PolicySettings()
    return best_settings, pd.DataFrame(rows)


def replay_policy(test_predictions: pd.DataFrame, settings: PolicySettings) -> tuple[pd.DataFrame, dict[str, float]]:
    trades = select_trades(test_predictions, settings)
    if not trades.empty and {"side", "realized_long_return", "realized_short_return"}.issubset(trades.columns):
        trades = trades.copy()
        trades["realized_net_return"] = np.where(
            trades["side"] == "SHORT",
            trades["realized_short_return"],
            trades["realized_long_return"],
        )
    return trades, evaluate_trades(trades)
