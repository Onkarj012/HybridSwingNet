#!/usr/bin/env python3
"""Replay selected V3 policy rows from a sweep CSV on another prediction window."""

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
    parser = argparse.ArgumentParser(description="Replay V3 policy configs from a sweep CSV")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-trades", type=int, default=0)
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


def parse_horizons(value: object) -> list[int]:
    if isinstance(value, float) and pd.isna(value):
        return []
    return [int(part) for part in str(value).split(",") if str(part).strip()]


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def selected_configs(path: Path, top_n: int, min_trades: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if min_trades:
        frame = frame[frame["trades"] >= min_trades]
    sort_cols = ["return_pct", "sharpe", "max_dd_pct"]
    frame = frame.sort_values(sort_cols, ascending=[False, False, True])
    key_cols = [
        "profile_name",
        "threshold",
        "horizons",
        "max_positions",
        "stop_atr",
        "target_atr",
        "trailing_stop_atr",
        "allow_shorts",
        "multi_horizon",
        "regime_policy",
        "dynamic_stocks",
    ]
    existing = [col for col in key_cols if col in frame.columns]
    return frame.drop_duplicates(existing).head(top_n).copy()


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

    rows: list[dict] = []
    configs = selected_configs(Path(args.configs), args.top_n, args.min_trades)
    print(f"replaying {len(configs)} configs", flush=True)
    for idx, cfg_row in enumerate(configs.itertuples(index=False), start=1):
        row = cfg_row._asdict()
        profile_name = row.get("profile_name", "conservative")
        base = PROFILES.get(str(profile_name), PROFILES["conservative"])
        threshold = float(row["threshold"])
        regime_policy = str(row.get("regime_policy", "off"))
        dynamic_stocks = int(row.get("dynamic_stocks", 0) or 0)
        profile = dataclasses.replace(
            base,
            max_positions=int(row["max_positions"]),
            min_p_up=threshold,
            max_p_up=1.0 - threshold,
            horizons=parse_horizons(row["horizons"]),
            stop_atr_mult=float(row["stop_atr"]),
            target_atr_mult=float(row["target_atr"]),
            trailing_stop_atr=float(row.get("trailing_stop_atr", 0.0) or 0.0),
            allow_shorts=bool_value(row["allow_shorts"]),
            one_position_per_symbol=not bool_value(row["multi_horizon"]),
            **INDIA_DELIVERY_COSTS,
        )
        backtester, meta = simulate_portfolio(
            profile,
            loader,
            trading_dates,
            preds_by_date,
            daily_data,
            regime_policy=regime_policy,
            regime_bullish=regime_bullish if regime_policy == "binary" else None,
            regime_scores=regime_scores if regime_policy == "graded" else None,
            regime_states=regime_states if regime_policy == "robust" else None,
            dynamic_stocks=dynamic_stocks,
            rolling_ranks=rolling_ranks if dynamic_stocks else None,
        )
        result = backtester.get_results()
        rows.append(
            {
                **row,
                "eval_return_pct": result.get("total_return_pct", 0.0),
                "eval_sharpe": result.get("sharpe_ratio", 0.0),
                "eval_max_dd_pct": result.get("max_drawdown_pct", 0.0),
                "eval_trades": result.get("total_trades", 0),
                "eval_win_rate": result.get("win_rate", 0.0),
                "eval_profit_factor": result.get("profit_factor", 0.0),
                "eval_regime_counts": json.dumps(meta.get("regime_counts", {}), sort_keys=True),
            }
        )
        if idx % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            print(f"replayed {idx} configs", flush=True)

    out = pd.DataFrame(rows).sort_values(["eval_return_pct", "eval_sharpe"], ascending=[False, False])
    out.to_csv(args.out_csv, index=False)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
