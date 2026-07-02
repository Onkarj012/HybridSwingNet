#!/usr/bin/env python3
"""Replay cached V3 predictions through regime-policy portfolio simulation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "src"))

from backtest_historical import PROFILES, Prediction
from backtest_v3 import compute_regime_bullish, compute_regime_scores, simulate_portfolio
from historical_data_loader import HistoricalDataLoader


def infer_window(predictions_path: Path) -> tuple[str, str]:
    summary_path = predictions_path.with_name("summary.json")
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        return summary["start"], summary["end"]

    frame = pd.read_csv(predictions_path, usecols=["date"])
    dates = pd.to_datetime(frame["date"])
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def add_atr14(frame: pd.DataFrame) -> pd.DataFrame:
    prev_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result = frame.copy()
    result["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    return result


def load_predictions(predictions_path: Path, daily_data: dict[str, pd.DataFrame]) -> List[Prediction]:
    frame = pd.read_csv(predictions_path)
    predictions: list[Prediction] = []
    for row in frame.itertuples(index=False):
        symbol = row.symbol
        date = row.date
        current_price = float(row.current_price)
        atr = current_price * 0.02
        symbol_data = daily_data.get(symbol)
        if symbol_data is not None:
            mask = symbol_data.index.strftime("%Y-%m-%d") == date
            if mask.any() and "atr_14" in symbol_data.columns:
                atr = float(symbol_data.loc[mask, "atr_14"].iloc[-1])

        predictions.append(
            Prediction(
                date=date,
                symbol=symbol,
                horizon=int(row.horizon),
                p_up=float(row.p_up),
                predicted_direction=str(row.predicted_dir),
                predicted_magnitude=float(getattr(row, "predicted_magnitude", 0.0)),
                confidence=float(row.confidence),
                current_price=current_price,
                vol_ref=0.015,
                atr=atr,
                actual_return_pct=float(row.actual_return_pct),
                actual_direction=str(row.actual_dir),
                is_correct=str(row.is_correct).lower() == "true",
            )
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="conservative")
    parser.add_argument("--policies", default="off,binary,graded")
    parser.add_argument("--regime-neutral-band", type=float, default=0.01)
    parser.add_argument("--regime-neutral-exposure", type=float, default=0.75)
    parser.add_argument("--regime-bear-exposure", type=float, default=0.0)
    parser.add_argument("--regime-neutral-threshold-shift", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold-shift", type=float, default=0.12)
    parser.add_argument("--slippage-pct", type=float, default=0.05)
    parser.add_argument("--stt-pct", type=float, default=0.1)
    parser.add_argument("--stamp-pct", type=float, default=0.015)
    parser.add_argument("--gst-pct", type=float, default=18.0)
    parser.add_argument("--exchange-pct", type=float, default=0.00345)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    start, end = infer_window(predictions_path)
    start = args.start or start
    end = args.end or end

    loader = HistoricalDataLoader(args.data_dir)
    raw_pred_frame = pd.read_csv(predictions_path, usecols=["symbol"])
    symbols = sorted(raw_pred_frame["symbol"].unique().tolist())
    warmup = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    daily_data = loader.get_daily_data_batch(symbols, start=warmup, end=data_end)
    daily_data = {symbol: add_atr14(frame) for symbol, frame in daily_data.items()}

    predictions = load_predictions(predictions_path, daily_data)
    preds_by_date: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        preds_by_date.setdefault(prediction.date, []).append(prediction)

    ref_symbol = max(daily_data, key=lambda symbol: len(daily_data[symbol]))
    trading_dates = loader.get_trading_dates(ref_symbol, start=start, end=end)
    profile = dataclasses.replace(
        PROFILES[args.profile],
        slippage_pct=args.slippage_pct,
        stt_pct=args.stt_pct,
        stamp_pct=args.stamp_pct,
        gst_pct=args.gst_pct,
        exchange_pct=args.exchange_pct,
    )

    regime_bullish = compute_regime_bullish(trading_dates, daily_data, symbols)
    regime_scores = compute_regime_scores(trading_dates, daily_data, symbols)

    rows = []
    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        backtester, meta = simulate_portfolio(
            profile,
            loader,
            trading_dates,
            preds_by_date,
            daily_data,
            regime_policy=policy,
            regime_bullish=regime_bullish,
            regime_scores=regime_scores,
            regime_neutral_band=args.regime_neutral_band,
            neutral_exposure=args.regime_neutral_exposure,
            bear_exposure=args.regime_bear_exposure,
            neutral_threshold_shift=args.regime_neutral_threshold_shift,
            bear_threshold_shift=args.regime_bear_threshold_shift,
        )
        result = backtester.get_results()
        rows.append(
            {
                "policy": policy,
                "return_pct": result.get("total_return_pct", 0.0),
                "sharpe": result.get("sharpe_ratio", 0.0),
                "max_dd_pct": result.get("max_drawdown_pct", 0.0),
                "trades": result.get("total_trades", 0),
                "win_rate": result.get("win_rate", 0.0),
                "profit_factor": result.get("profit_factor", 0.0),
                "skipped_days": meta["regime_skipped_days"],
                "bull_days": meta["regime_counts"]["bull"],
                "neutral_days": meta["regime_counts"]["neutral"],
                "bear_days": meta["regime_counts"]["bear"],
            }
        )

    result_frame = pd.DataFrame(rows)
    print(result_frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
