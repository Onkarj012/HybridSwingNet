#!/usr/bin/env python3
"""Benchmark rows for the walk-forward OOS window, at the same full-cost engine.

Computes, over the same concatenated OOS window as the walk-forward strategy
(2021-01-01 -> 2026-06-25, full India delivery costs throughout):
  1. Equal-weight buy-and-hold (Nifty 100 working universe)
  2. SMA(10/50) crossover portfolio, routed through the same PortfolioBacktester
     execution engine (same cost model) via synthetic long-only predictions
  3. Random-entry, turnover approximately matched to the walk-forward strategy,
     averaged over multiple seeds

Per docs/PREREGISTERED_STRATEGY_PROTOCOL.md Phase 4.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from backtest_historical import PROFILES, Prediction
from backtest_v3 import INDIA_DELIVERY_COSTS, build_v3_feature_data, simulate_portfolio
from historical_data_loader import HistoricalDataLoader
from stockxpert.utils import set_seed
from stockxpert.v3.config import load_v3_config

WINDOW_START = "2021-01-01"
# Capped at data/nifty500 minute-bar coverage end (same execution data source as the
# walk-forward run) - actual last trading day available is 2026-04-08, not the nominal
# 2026-06-25 H1 end. Matches runs/walkforward_2021_2026H1/walkforward_summary.json.
WINDOW_END = "2026-04-08"


def atr_lookup(feature_data: dict[str, pd.DataFrame]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for symbol, frame in feature_data.items():
        for dt, row in frame.iterrows():
            atr = row.get("atr_14", np.nan)
            close = row.get("Close", np.nan)
            if pd.isna(atr) or atr <= 0:
                atr = float(close) * 0.02
            lookup[(symbol, dt.strftime("%Y-%m-%d"))] = float(atr)
    return lookup


def buy_and_hold(daily_data: dict[str, pd.DataFrame], start: str, end: str) -> dict:
    """Equal-weight buy-and-hold across the working universe; one entry + one exit cost."""
    cost_pct = sum(INDIA_DELIVERY_COSTS.values()) + 0.03  # + brokerage
    prices = {}
    for sym, df in daily_data.items():
        window = df[(df.index >= start) & (df.index <= end)]
        if len(window) < 2:
            continue
        prices[sym] = window["Close"]

    if not prices:
        raise RuntimeError("No symbols with sufficient data for buy-and-hold window")

    aligned = pd.DataFrame(prices).sort_index().ffill().dropna(how="all")
    norm = aligned / aligned.iloc[0]
    equal_weight_curve = norm.mean(axis=1) * 100000.0
    equal_weight_curve.iloc[0] *= (1 - cost_pct / 100)  # entry cost
    equal_weight_curve.iloc[-1] *= (1 - cost_pct / 100)  # exit cost

    values = equal_weight_curve.to_numpy()
    daily_returns = np.diff(values) / values[:-1]
    total_return_pct = (values[-1] / 100000.0 - 1) * 100
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252))
    peak = np.maximum.accumulate(values)
    max_dd_pct = float(np.max((peak - values) / peak) * 100)

    return {
        "name": "Nifty100 equal-weight buy-and-hold",
        "n_symbols": len(prices),
        "total_return_pct": float(total_return_pct),
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "n_days": len(values),
    }


def sma_crossover_predictions(daily_data: dict[str, pd.DataFrame], start: str, end: str, atrs) -> list[Prediction]:
    preds: list[Prediction] = []
    for sym, df in daily_data.items():
        window = df[(df.index >= start) & (df.index <= end)].copy()
        if len(window) < 55:
            continue
        window["sma10"] = window["Close"].rolling(10).mean()
        window["sma50"] = window["Close"].rolling(50).mean()
        bullish = (window["sma10"] > window["sma50"]) & (window["sma10"].shift(1) <= window["sma50"].shift(1))
        for dt in window.index[bullish]:
            date_str = dt.strftime("%Y-%m-%d")
            close = float(window.loc[dt, "Close"])
            preds.append(
                Prediction(
                    date=date_str,
                    symbol=sym,
                    horizon=10,
                    p_up=0.99,
                    predicted_direction="UP",
                    predicted_magnitude=0.0,
                    confidence=0.99,
                    current_price=close,
                    vol_ref=0.015,
                    atr=atrs.get((sym, date_str), close * 0.02),
                )
            )
    return preds


def random_entry_predictions(
    daily_data: dict[str, pd.DataFrame], start: str, end: str, atrs, seed: int, entry_rate: float
) -> list[Prediction]:
    rng = np.random.default_rng(seed)
    preds: list[Prediction] = []
    for sym, df in daily_data.items():
        window = df[(df.index >= start) & (df.index <= end)]
        for dt in window.index:
            if rng.random() < entry_rate:
                date_str = dt.strftime("%Y-%m-%d")
                close = float(window.loc[dt, "Close"])
                preds.append(
                    Prediction(
                        date=date_str,
                        symbol=sym,
                        horizon=10,
                        p_up=float(rng.uniform(0.5, 1.0)),
                        predicted_direction="UP",
                        predicted_magnitude=0.0,
                        confidence=0.6,
                        current_price=close,
                        vol_ref=0.015,
                        atr=atrs.get((sym, date_str), close * 0.02),
                    )
                )
    return preds


def run_engine_strategy(
    predictions: list[Prediction],
    loader: HistoricalDataLoader,
    reference_symbol: str,
    daily_data: dict,
    start: str,
    end: str,
    top_n: int,
    threshold: float,
) -> dict:
    preds_by_date: dict[str, list[Prediction]] = {}
    for p in predictions:
        preds_by_date.setdefault(p.date, []).append(p)

    profile = dataclasses.replace(
        PROFILES["conservative"],
        max_positions=top_n,
        min_p_up=threshold,
        max_p_up=1.0 - threshold,
        horizons=[10],
        allow_shorts=False,
        one_position_per_symbol=True,
        trailing_stop_atr=0.0,
        **INDIA_DELIVERY_COSTS,
    )
    trading_dates = loader.get_trading_dates(reference_symbol, start=start, end=end)
    backtester, _ = simulate_portfolio(
        profile, loader, trading_dates, preds_by_date, daily_data, regime_policy="off"
    )
    result = backtester.get_results()
    return {
        "total_return_pct": result.get("total_return_pct", 0.0),
        "sharpe": result.get("sharpe_ratio", 0.0),
        "max_dd_pct": result.get("max_drawdown_pct", 0.0),
        "trades": result.get("total_trades", 0),
        "win_rate": result.get("win_rate", 0.0),
        "profit_factor": result.get("profit_factor", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward window benchmark rows")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--walkforward-total-trades", type=int, required=True,
                         help="Total trades in the walk-forward strategy over the OOS window, for matched-turnover calibration")
    parser.add_argument("--random-seeds", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)
    cfg = load_v3_config(run_dir / "config.yaml")
    set_seed(cfg.run.seed)

    loader = HistoricalDataLoader(args.data_dir)
    available = loader.get_available_model_symbols(list(cfg.data.symbols))
    cfg.data.symbols = available
    cfg.data.end = WINDOW_END

    print("Building feature data for benchmark window...", flush=True)
    feature_data = build_v3_feature_data(cfg, args.data_dir)
    feature_data = {sym: df for sym, df in feature_data.items() if sym in available}
    daily_data = {
        sym: df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for sym, df in feature_data.items()
        if all(col in df.columns for col in ("Open", "High", "Low", "Close"))
    }
    atrs = atr_lookup(feature_data)
    reference_symbol = available[0]

    results = {}

    print("Benchmark A: buy-and-hold...", flush=True)
    results["buy_and_hold"] = buy_and_hold(daily_data, WINDOW_START, WINDOW_END)

    print("Benchmark B: SMA(10/50) crossover...", flush=True)
    sma_preds = sma_crossover_predictions(daily_data, WINDOW_START, WINDOW_END, atrs)
    results["sma_crossover"] = {
        "name": "SMA(10/50) crossover portfolio, top-3, same execution engine",
        **run_engine_strategy(sma_preds, loader, reference_symbol, daily_data, WINDOW_START, WINDOW_END, top_n=3, threshold=0.5),
    }

    print("Benchmark C: random-entry, matched turnover...", flush=True)
    n_days = len(loader.get_trading_dates(reference_symbol, start=WINDOW_START, end=WINDOW_END))
    n_symbols = len(daily_data)
    # crude calibration: entries needed roughly = target trades; each day-symbol has
    # `entry_rate` chance of firing a candidate, backtester then filters to top-3/day.
    target_trades = args.walkforward_total_trades
    entry_rate = min(0.5, max(0.001, target_trades / max(1, n_days * n_symbols) * 8))
    seed_results = []
    for seed in range(args.random_seeds):
        rnd_preds = random_entry_predictions(daily_data, WINDOW_START, WINDOW_END, atrs, seed, entry_rate)
        seed_results.append(
            run_engine_strategy(rnd_preds, loader, reference_symbol, daily_data, WINDOW_START, WINDOW_END, top_n=3, threshold=0.5)
        )
    avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
    results["random_matched_turnover"] = {
        "name": f"Random-entry, {args.random_seeds}-seed average, target_trades~{target_trades}",
        "entry_rate_used": entry_rate,
        "per_seed": seed_results,
        **avg,
    }

    (out_dir / "walkforward_benchmarks.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
