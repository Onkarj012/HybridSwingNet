#!/usr/bin/env python3
"""Recompute bootstrap Sharpe CI + deflated Sharpe for the canonical 2025 run.

Per docs/CANONICAL_RESULTS.md §12 item 10 (still needed before submission) and
docs/PREREGISTERED_STRATEGY_PROTOCOL.md Phase 4. Independent of the walk-forward
pipeline; uses the existing canonical run artifact directly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import bootstrap_sharpe, deflated_sharpe  # noqa: E402

CANONICAL_RUN = Path("runs/v3_canonical_regen_2025_binary_fullcost_legacy_exec")
N_CONFIGS_K = 4  # from CANONICAL_RESULTS.md §8.2: the 4 V3 configs evaluated


def main() -> None:
    equity = pd.read_csv(CANONICAL_RUN / "equity_curve_conservative.csv")
    values = equity["Portfolio_Value"].to_numpy(dtype=float)
    daily_returns = np.diff(values) / values[:-1]

    boot = bootstrap_sharpe(daily_returns, n_bootstrap=10000, ci=0.95)
    point_sharpe = boot["sharpe_point"]

    dsr = deflated_sharpe(point_sharpe, n_configs=N_CONFIGS_K, n_observations=len(daily_returns))

    skew = float(pd.Series(daily_returns).skew())
    kurt = float(pd.Series(daily_returns).kurt() + 3)  # pandas kurt() is excess; DSR convention uses raw

    result = {
        "source_run": str(CANONICAL_RUN),
        "n_daily_observations": len(daily_returns),
        "sharpe_point": point_sharpe,
        "bootstrap_95ci_low": boot["ci_low"],
        "bootstrap_95ci_high": boot["ci_high"],
        "bootstrap_std": boot["bootstrap_std"],
        "n_configs_k": N_CONFIGS_K,
        "deflated_sharpe": round(dsr, 3),
        "sharpe_exceeds_dsr_threshold": bool(point_sharpe > dsr),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "method": "Bailey & Lopez de Prado (2014); bootstrap n=10000, seed=42, from CANONICAL_RESULTS.md canonical run daily equity curve",
    }

    out_path = Path("reports/canonical_2025_stats_recomputed.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
