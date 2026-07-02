#!/usr/bin/env python3
"""Block-bootstrap Sharpe CI + deflated Sharpe for the walk-forward OOS curve.

Per docs/PREREGISTERED_STRATEGY_PROTOCOL.md Phase 4. Uses a moving-block
bootstrap (not i.i.d.) because the concatenated daily-return series has
within-fold autocorrelation from ATR-horizon holding periods.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WALKFORWARD_DIR = Path("runs/walkforward_2021_2026H1")
BLOCK_SIZE = 20  # ~1 trading month
N_BOOTSTRAP = 10000
GRID_SIZE_PER_FOLD = 144
N_FOLDS = 6


def moving_block_bootstrap_sharpe(returns: np.ndarray, block_size: int, n_bootstrap: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks_needed = math.ceil(n / block_size)
    max_start = n - block_size
    sharpes = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        sample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        std = np.std(sample, ddof=1)
        if std > 0:
            sharpes.append(float(np.mean(sample) / std * np.sqrt(252)))
    sharpes = np.array(sharpes)
    point = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
    alpha = 0.025
    return {
        "sharpe_point": round(point, 3),
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
        "ci_low": round(float(np.percentile(sharpes, alpha * 100)), 3),
        "ci_high": round(float(np.percentile(sharpes, (1 - alpha) * 100)), 3),
        "bootstrap_std": round(float(np.std(sharpes)), 3),
    }


def deflated_sharpe(point_sharpe: float, n_configs: int, n_observations: int, p: float = 0.05) -> float:
    if n_observations <= 0 or n_configs <= 0:
        return point_sharpe
    deflation = 1 - (n_configs / n_observations) * math.log(1 / p)
    return point_sharpe * deflation


def main() -> None:
    concat = pd.read_csv(WALKFORWARD_DIR / "walkforward_concatenated_daily_returns.csv")
    returns = concat["daily_return"].to_numpy(dtype=float)
    summary = json.loads((WALKFORWARD_DIR / "walkforward_summary.json").read_text())

    boot = moving_block_bootstrap_sharpe(returns, BLOCK_SIZE, N_BOOTSTRAP)

    # k=1 for the walk-forward meta-strategy itself (selection was internal/automatic,
    # pre-specified per fold, no human re-selection after seeing OOS results). Grid size
    # disclosed anyway for transparency.
    dsr_k1 = deflated_sharpe(boot["sharpe_point"], n_configs=1, n_observations=len(returns))
    # Also report the deflation if the full per-fold selection grid were treated as k
    # (upper-bound conservative disclosure, not used as the headline).
    dsr_disclosed_grid = deflated_sharpe(
        boot["sharpe_point"], n_configs=GRID_SIZE_PER_FOLD, n_observations=len(returns)
    )

    result = {
        "n_daily_observations": len(returns),
        "window": {"start": concat["date"].iloc[0], "end": concat["date"].iloc[-1]},
        "concatenated_summary": summary["concatenated_oos"],
        "block_bootstrap_sharpe": boot,
        "dsr_k1_meta_strategy": round(dsr_k1, 3),
        "dsr_k_equals_grid_144_conservative_disclosure": round(dsr_disclosed_grid, 3),
        "grid_size_per_fold": GRID_SIZE_PER_FOLD,
        "n_folds": N_FOLDS,
        "total_backtests_run": GRID_SIZE_PER_FOLD * N_FOLDS * 4 + N_FOLDS,
        "note": "Point Sharpe is negative; deflation is reported for completeness/transparency but is not the operative finding here.",
    }

    out_path = Path("reports/walkforward_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
