"""V4 after-cost utility trade selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PolicySettings:
    min_expected_return_bps: float = 0.0
    max_positions: int = 5
    allow_shorts: bool = False
    risk_aversion: float = 0.5
    long_only: bool = True


REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "horizon",
    "expected_long_return",
    "expected_short_return",
    "downside_risk",
    "cost_bps",
}


def score_candidates(predictions: pd.DataFrame, settings: PolicySettings) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing V4 policy columns: {sorted(missing)}")
    out = predictions.copy()
    cost = out["cost_bps"].astype(float) / 10_000.0
    out["long_utility"] = out["expected_long_return"].astype(float) - cost - settings.risk_aversion * out["downside_risk"].astype(float)
    out["short_utility"] = out["expected_short_return"].astype(float) - cost - settings.risk_aversion * out["downside_risk"].astype(float)
    if settings.allow_shorts and not settings.long_only:
        out["side"] = out.apply(lambda row: "SHORT" if row["short_utility"] > row["long_utility"] else "LONG", axis=1)
        out["utility"] = out[["long_utility", "short_utility"]].max(axis=1)
    else:
        out["side"] = "LONG"
        out["utility"] = out["long_utility"]
    return out


def select_trades(predictions: pd.DataFrame, settings: PolicySettings) -> pd.DataFrame:
    """Select trades only when expected after-cost utility is positive."""
    scored = score_candidates(predictions, settings)
    min_utility = settings.min_expected_return_bps / 10_000.0
    scored = scored[scored["utility"] > min_utility]
    if settings.long_only or not settings.allow_shorts:
        scored = scored[scored["side"] == "LONG"]
    scored = scored.sort_values(["date", "symbol", "utility"], ascending=[True, True, False])
    scored = scored.groupby(["date", "symbol"], group_keys=False).head(1)
    scored = scored.sort_values(["date", "utility"], ascending=[True, False])
    return scored.groupby("date", group_keys=False).head(settings.max_positions).reset_index(drop=True)
