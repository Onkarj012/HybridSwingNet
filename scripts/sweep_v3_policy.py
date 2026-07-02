#!/usr/bin/env python3
"""Sweep V3 portfolio policies from saved prediction CSVs."""

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
from backtest_v3 import (
    INDIA_DELIVERY_COSTS,
    build_rolling_ranks,
    build_v3_feature_data,
    compute_regime_bullish,
    compute_regime_scores,
    compute_robust_regime_states,
    simulate_portfolio,
)
from historical_data_loader import HistoricalDataLoader
from stockxpert.utils import set_seed
from stockxpert.v3.config import load_v3_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep V3 policy variants from saved predictions")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--top-json", default=None)
    parser.add_argument(
        "--grid",
        choices=["narrow", "broad"],
        default="narrow",
        help="Sweep size. broad expands profile, threshold, horizon, position, stop/target, regime, and dynamic-stock knobs.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke-testing the generated grid.")
    return parser.parse_args()


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


def load_predictions(path: Path, atrs: dict[tuple[str, str], float]) -> list[Prediction]:
    frame = pd.read_csv(path)
    predictions: list[Prediction] = []
    for row in frame.itertuples(index=False):
        date = str(row.date)
        symbol = str(row.symbol)
        p_up = float(row.p_up)
        actual_dir = "" if pd.isna(row.actual_dir) else str(row.actual_dir)
        predictions.append(
            Prediction(
                date=date,
                symbol=symbol,
                horizon=int(row.horizon),
                p_up=p_up,
                predicted_direction=str(row.predicted_dir),
                predicted_magnitude=0.0,
                confidence=float(row.confidence),
                current_price=float(row.current_price),
                vol_ref=0.015,
                atr=atrs.get((symbol, date), float(row.current_price) * 0.02),
                actual_return_pct=float(row.actual_return_pct) if not pd.isna(row.actual_return_pct) else 0.0,
                actual_direction=actual_dir,
                is_correct=bool(row.is_correct),
            )
        )
    return predictions


def config_grid(grid: str) -> list[dict]:
    if grid == "narrow":
        profile_names = ["conservative"]
        thresholds = [0.62, 0.65, 0.68]
        horizon_sets = {
            "h5710": [5, 7, 10],
            "h710": [7, 10],
            "h10": [10],
        }
        max_positions_set = [3]
        stop_targets = [(1.5, 2.5), (1.5, 3.0), (2.0, 3.0)]
        regime_policies = ["off", "binary", "robust"]
        dynamic_stocks_set = [0]
        trailing_stops = [0.0]
    else:
        profile_names = ["conservative", "moderate", "aggressive", "intraday", "swing", "max_diversified"]
        thresholds = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]
        horizon_sets = {
            "h1": [1],
            "h3": [3],
            "h5": [5],
            "h7": [7],
            "h10": [10],
            "h13": [1, 3],
            "h35": [3, 5],
            "h57": [5, 7],
            "h710": [7, 10],
            "h357": [3, 5, 7],
            "h5710": [5, 7, 10],
            "hall": [1, 3, 5, 7, 10],
        }
        max_positions_set = [1, 2, 3, 5, 8]
        stop_targets = [
            (0.75, 1.0),
            (1.0, 1.5),
            (1.0, 2.0),
            (1.25, 2.0),
            (1.5, 2.0),
            (1.5, 2.5),
            (1.5, 3.0),
            (2.0, 3.0),
            (2.0, 4.0),
        ]
        regime_policies = ["off", "binary", "graded", "robust"]
        dynamic_stocks_set = [0, 15, 30]
        trailing_stops = [0.0, 1.0]

    records = []
    for profile_name in profile_names:
        for threshold in thresholds:
            for horizon_name, horizons in horizon_sets.items():
                for max_positions in max_positions_set:
                    for stop_atr, target_atr in stop_targets:
                        for trailing_stop_atr in trailing_stops:
                            for allow_shorts in [True, False]:
                                for multi_horizon in [True, False]:
                                    for regime_policy in regime_policies:
                                        for dynamic_stocks in dynamic_stocks_set:
                                            if dynamic_stocks and regime_policy == "off":
                                                # Dynamic filters are a production/risk overlay; keep the off-regime
                                                # branch focused on pure signal/execution settings.
                                                continue
                                            if len(horizons) == 1 and multi_horizon:
                                                continue
                                            if target_atr <= stop_atr:
                                                continue
                                            if profile_name == "intraday" and any(h > 3 for h in horizons):
                                                continue
                                            if profile_name == "swing" and any(h < 5 for h in horizons):
                                                continue
                                            if threshold >= 0.68 and profile_name == "aggressive":
                                                continue
                                            if threshold <= 0.58 and profile_name == "conservative":
                                                continue
                                            if trailing_stop_atr and target_atr < 2.0:
                                                continue
                                            if not allow_shorts and threshold <= 0.55:
                                                continue
                                records.append(
                                    {
                                        "profile_name": profile_name,
                                        "threshold": threshold,
                                        "horizon_name": horizon_name,
                                        "horizons": horizons,
                                        "max_positions": max_positions,
                                        "stop_atr": stop_atr,
                                        "target_atr": target_atr,
                                        "trailing_stop_atr": trailing_stop_atr,
                                        "allow_shorts": allow_shorts,
                                        "multi_horizon": multi_horizon,
                                        "regime_policy": regime_policy,
                                        "dynamic_stocks": dynamic_stocks,
                                    }
                                )
    return records


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_v3_config(run_dir / "config.yaml")
    set_seed(cfg.run.seed)
    if args.end > cfg.data.end:
        cfg.data.end = args.end

    loader = HistoricalDataLoader(args.data_dir)
    available = loader.get_available_model_symbols(list(cfg.data.symbols))
    cfg.data.symbols = available

    feature_data = build_v3_feature_data(cfg, args.data_dir)
    feature_data = {sym: df for sym, df in feature_data.items() if sym in available}
    daily_data = {
        sym: df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for sym, df in feature_data.items()
        if all(col in df.columns for col in ("Open", "High", "Low", "Close"))
    }

    trading_dates = loader.get_trading_dates(available[0], start=args.start, end=args.end)
    predictions = load_predictions(Path(args.predictions), atr_lookup(feature_data))
    preds_by_date: dict[str, list[Prediction]] = {}
    for pred in predictions:
        preds_by_date.setdefault(pred.date, []).append(pred)

    regime_bullish = compute_regime_bullish(trading_dates, daily_data, available)
    regime_states = compute_robust_regime_states(trading_dates, daily_data, available)
    regime_scores = compute_regime_scores(trading_dates, daily_data, available)
    rolling_ranks = build_rolling_ranks(predictions, horizon=1)

    def save_rows(rows: list[dict]) -> pd.DataFrame:
        out = pd.DataFrame(rows).sort_values(
            ["return_pct", "sharpe", "max_dd_pct"], ascending=[False, False, True]
        )
        out.to_csv(args.out_csv, index=False)
        if args.top_json:
            Path(args.top_json).write_text(out.head(20).to_json(orient="records", indent=2))
        return out

    rows = []
    grid_rows = config_grid(args.grid)
    if args.limit > 0:
        grid_rows = grid_rows[: args.limit]
    print(f"sweeping {len(grid_rows)} configs ({args.grid})", flush=True)
    try:
        for idx, cfg_row in enumerate(grid_rows, start=1):
            base = PROFILES[cfg_row["profile_name"]]
            profile = dataclasses.replace(
                base,
                max_positions=cfg_row["max_positions"],
                min_p_up=cfg_row["threshold"],
                max_p_up=1.0 - cfg_row["threshold"],
                horizons=cfg_row["horizons"],
                stop_atr_mult=cfg_row["stop_atr"],
                target_atr_mult=cfg_row["target_atr"],
                trailing_stop_atr=cfg_row["trailing_stop_atr"],
                allow_shorts=cfg_row["allow_shorts"],
                one_position_per_symbol=not cfg_row["multi_horizon"],
                **INDIA_DELIVERY_COSTS,
            )
            backtester, meta = simulate_portfolio(
                profile,
                loader,
                trading_dates,
                preds_by_date,
                daily_data,
                regime_policy=cfg_row["regime_policy"],
                regime_bullish=regime_bullish if cfg_row["regime_policy"] == "binary" else None,
                regime_scores=regime_scores if cfg_row["regime_policy"] == "graded" else None,
                regime_states=regime_states if cfg_row["regime_policy"] == "robust" else None,
                dynamic_stocks=cfg_row["dynamic_stocks"],
                rolling_ranks=rolling_ranks if cfg_row["dynamic_stocks"] else None,
            )
            result = backtester.get_results()
            rows.append(
                {
                    **{k: v for k, v in cfg_row.items() if k != "horizons"},
                    "horizons": ",".join(str(h) for h in cfg_row["horizons"]),
                    "return_pct": result.get("total_return_pct", 0.0),
                    "sharpe": result.get("sharpe_ratio", 0.0),
                    "max_dd_pct": result.get("max_drawdown_pct", 0.0),
                    "trades": result.get("total_trades", 0),
                    "win_rate": result.get("win_rate", 0.0),
                    "profit_factor": result.get("profit_factor", 0.0),
                    "regime_counts": json.dumps(meta.get("regime_counts", {}), sort_keys=True),
                }
            )
            if idx % 250 == 0:
                save_rows(rows)
                print(f"swept {idx} configs", flush=True)
    finally:
        if rows:
            out = save_rows(rows)
        else:
            out = pd.DataFrame()

    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
