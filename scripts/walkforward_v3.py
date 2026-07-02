#!/usr/bin/env python3
"""Pre-registered walk-forward strategy selection + OOS trading.

Implements docs/PREREGISTERED_STRATEGY_PROTOCOL.md exactly: 144-config grid,
worst-selection-year net-return selection rule (min 30 trades/fold, Sharpe
tie-break), 6-fold rolling 3y-select -> 1y-trade schedule, full India delivery
cost model throughout. One evaluation, no re-runs.
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
from backtest_v3 import (
    INDIA_DELIVERY_COSTS,
    build_v3_feature_data,
    compute_regime_bullish,
    simulate_portfolio,
)
from historical_data_loader import HistoricalDataLoader
from stockxpert.utils import set_seed
from stockxpert.v3.config import load_v3_config

MIN_TRADES_PER_FOLD = 30

FOLDS = [
    {"id": 1, "sel_start": "2018-01-01", "sel_end": "2020-12-31", "trade_start": "2021-01-01", "trade_end": "2021-12-31"},
    {"id": 2, "sel_start": "2019-01-01", "sel_end": "2021-12-31", "trade_start": "2022-01-01", "trade_end": "2022-12-31"},
    {"id": 3, "sel_start": "2020-01-01", "sel_end": "2022-12-31", "trade_start": "2023-01-01", "trade_end": "2023-12-31"},
    {"id": 4, "sel_start": "2021-01-01", "sel_end": "2023-12-31", "trade_start": "2024-01-01", "trade_end": "2024-12-31"},
    {"id": 5, "sel_start": "2022-01-01", "sel_end": "2024-12-31", "trade_start": "2025-01-01", "trade_end": "2025-12-31"},
    {"id": 6, "sel_start": "2023-01-01", "sel_end": "2025-12-31", "trade_start": "2026-01-01", "trade_end": "2026-06-25"},
]


def selection_years(sel_start: str, sel_end: str) -> list[tuple[str, str]]:
    start_year = int(sel_start[:4])
    end_year = int(sel_end[:4])
    years = []
    for y in range(start_year, end_year + 1):
        y_start = f"{y}-01-01" if y != start_year else sel_start
        y_end = f"{y}-12-31" if y != end_year else sel_end
        years.append((y_start, y_end))
    return years


def build_grid() -> list[dict]:
    horizon_sets = {"H5": [5], "H7": [7], "H10": [10]}
    top_ns = [1, 2, 3]
    thresholds = [0.55, 0.60, 0.65, 0.70]
    regimes = ["off", "binary"]
    sides = [False, True]  # allow_shorts

    records = []
    for h_name, horizons in horizon_sets.items():
        for top_n in top_ns:
            for threshold in thresholds:
                for regime in regimes:
                    for allow_shorts in sides:
                        records.append(
                            {
                                "horizon_name": h_name,
                                "horizons": horizons,
                                "top_n": top_n,
                                "threshold": threshold,
                                "regime": regime,
                                "allow_shorts": allow_shorts,
                            }
                        )
    assert len(records) == 144, f"expected 144 configs, got {len(records)}"
    return records


def make_profile(cfg_row: dict) -> object:
    base = PROFILES["conservative"]
    return dataclasses.replace(
        base,
        max_positions=cfg_row["top_n"],
        min_p_up=cfg_row["threshold"],
        max_p_up=1.0 - cfg_row["threshold"],
        horizons=cfg_row["horizons"],
        allow_shorts=cfg_row["allow_shorts"],
        one_position_per_symbol=True,
        trailing_stop_atr=0.0,
        **INDIA_DELIVERY_COSTS,
    )


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


def run_config(
    cfg_row: dict,
    loader: HistoricalDataLoader,
    reference_symbol: str,
    preds_by_date: dict,
    daily_data: dict,
    regime_bullish: dict,
    start: str,
    end: str,
) -> tuple[object, dict]:
    profile = make_profile(cfg_row)
    trading_dates = loader.get_trading_dates(reference_symbol, start=start, end=end)
    backtester, meta = simulate_portfolio(
        profile,
        loader,
        trading_dates,
        preds_by_date,
        daily_data,
        regime_policy=cfg_row["regime"],
        regime_bullish=regime_bullish if cfg_row["regime"] == "binary" else None,
    )
    return backtester, backtester.get_results()


def evaluate_fold_grid(
    grid: list[dict],
    fold: dict,
    loader: HistoricalDataLoader,
    reference_symbol: str,
    preds_by_date: dict,
    daily_data: dict,
    regime_bullish: dict,
) -> pd.DataFrame:
    years = selection_years(fold["sel_start"], fold["sel_end"])
    rows = []
    for cfg_row in grid:
        year_returns = []
        total_trades = 0
        for y_start, y_end in years:
            _, result = run_config(
                cfg_row, loader, reference_symbol, preds_by_date, daily_data, regime_bullish, y_start, y_end
            )
            year_returns.append(result.get("total_return_pct", 0.0))
            total_trades += result.get("total_trades", 0)

        _, cont_result = run_config(
            cfg_row, loader, reference_symbol, preds_by_date, daily_data, regime_bullish,
            fold["sel_start"], fold["sel_end"],
        )

        rows.append(
            {
                **{k: v for k, v in cfg_row.items() if k != "horizons"},
                "horizons": ",".join(str(h) for h in cfg_row["horizons"]),
                "year_returns_pct": json.dumps([round(r, 4) for r in year_returns]),
                "worst_year_return_pct": min(year_returns),
                "total_trades_3y": total_trades,
                "continuous_3y_return_pct": cont_result.get("total_return_pct", 0.0),
                "continuous_3y_sharpe": cont_result.get("sharpe_ratio", 0.0),
                "eligible": total_trades >= MIN_TRADES_PER_FOLD,
            }
        )
    return pd.DataFrame(rows)


def select_winner(grid_df: pd.DataFrame) -> pd.Series:
    eligible = grid_df[grid_df["eligible"]]
    pool = eligible if len(eligible) else grid_df.sort_values("total_trades_3y", ascending=False).head(5)
    ranked = pool.sort_values(
        ["worst_year_return_pct", "continuous_3y_sharpe"], ascending=[False, False]
    )
    return ranked.iloc[0]


def cfg_row_from_series(row: pd.Series) -> dict:
    return {
        "horizon_name": row["horizon_name"],
        "horizons": [int(h) for h in str(row["horizons"]).split(",")],
        "top_n": int(row["top_n"]),
        "threshold": float(row["threshold"]),
        "regime": row["regime"],
        "allow_shorts": bool(row["allow_shorts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered V3 walk-forward pipeline")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)
    cfg = load_v3_config(run_dir / "config.yaml")
    set_seed(cfg.run.seed)

    loader = HistoricalDataLoader(args.data_dir)
    available = loader.get_available_model_symbols(list(cfg.data.symbols))
    cfg.data.symbols = available
    cfg.data.end = "2026-06-25"

    print("Building feature data (one pass, full 2018-2026 range)...", flush=True)
    feature_data = build_v3_feature_data(cfg, args.data_dir)
    feature_data = {sym: df for sym, df in feature_data.items() if sym in available}
    daily_data = {
        sym: df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for sym, df in feature_data.items()
        if all(col in df.columns for col in ("Open", "High", "Low", "Close"))
    }

    print("Loading predictions...", flush=True)
    predictions = load_predictions(Path(args.predictions), atr_lookup(feature_data))
    preds_by_date: dict[str, list[Prediction]] = {}
    for pred in predictions:
        preds_by_date.setdefault(pred.date, []).append(pred)

    reference_symbol = available[0]
    print("Computing binary regime over full range...", flush=True)
    full_dates = loader.get_trading_dates(reference_symbol, start="2018-01-01", end="2026-06-25")
    regime_bullish = compute_regime_bullish(full_dates, daily_data, available)

    grid = build_grid()
    print(f"Grid size: {len(grid)} configs/fold", flush=True)

    fold_summaries = []
    concatenated_daily_returns: list[tuple[str, float]] = []

    for fold in FOLDS:
        fold_id = fold["id"]
        print(f"\n=== Fold {fold_id}: select {fold['sel_start']}..{fold['sel_end']} -> trade {fold['trade_start']}..{fold['trade_end']} ===", flush=True)

        grid_df = evaluate_fold_grid(grid, fold, loader, reference_symbol, preds_by_date, daily_data, regime_bullish)
        grid_df.to_csv(out_dir / f"fold{fold_id}_selection_grid.csv", index=False)

        winner_row = select_winner(grid_df)
        winner_cfg = cfg_row_from_series(winner_row)
        print(f"Fold {fold_id} winner: {winner_cfg} (worst_year_return={winner_row['worst_year_return_pct']:.2f}%, eligible={bool(winner_row['eligible'])})", flush=True)

        backtester, trade_result = run_config(
            winner_cfg, loader, reference_symbol, preds_by_date, daily_data, regime_bullish,
            fold["trade_start"], fold["trade_end"],
        )

        trade_log_rows = [
            {
                "fold": fold_id,
                "Symbol": t.symbol,
                "Side": t.side,
                "Horizon": t.horizon,
                "P_Up": t.p_up,
                "Entry_Date": t.entry_date,
                "Entry_Price": t.entry_price,
                "Exit_Date": t.exit_date,
                "Exit_Price": t.exit_price,
                "Exit_Reason": t.exit_reason,
                "Entry_Cost": t.entry_cost,
                "Exit_Cost": t.exit_cost,
                "Gross_PnL": t.gross_pnl,
                "Net_PnL": t.net_pnl,
                "Accounting_Version": t.accounting_version,
                "PnL": t.pnl,
                "PnL_Pct": t.pnl_pct,
            }
            for t in backtester.closed_trades
        ]
        pd.DataFrame(trade_log_rows).to_csv(out_dir / f"fold{fold_id}_trade_log.csv", index=False)

        equity_rows = [{"date": d, "equity": v} for d, v in backtester.daily_equity]
        pd.DataFrame(equity_rows).to_csv(out_dir / f"fold{fold_id}_equity_curve.csv", index=False)

        equity_values = [v for _, v in backtester.daily_equity]
        daily_returns = list(np.diff(equity_values) / np.array(equity_values[:-1])) if len(equity_values) > 1 else []
        fold_dates = [d for d, _ in backtester.daily_equity][1:]
        concatenated_daily_returns.extend(zip(fold_dates, daily_returns))

        fold_summaries.append(
            {
                "fold": fold_id,
                "sel_start": fold["sel_start"],
                "sel_end": fold["sel_end"],
                "trade_start": fold["trade_start"],
                "trade_end": fold["trade_end"],
                "winner_config": winner_cfg,
                "winner_eligible": bool(winner_row["eligible"]),
                "winner_worst_selection_year_return_pct": float(winner_row["worst_year_return_pct"]),
                "winner_selection_total_trades_3y": int(winner_row["total_trades_3y"]),
                "trade_window_return_pct": trade_result.get("total_return_pct", 0.0),
                "trade_window_sharpe": trade_result.get("sharpe_ratio", 0.0),
                "trade_window_max_dd_pct": trade_result.get("max_drawdown_pct", 0.0),
                "trade_window_trades": trade_result.get("total_trades", 0),
                "trade_window_win_rate": trade_result.get("win_rate", 0.0),
                "trade_window_profit_factor": trade_result.get("profit_factor", 0.0),
            }
        )

    # Concatenated OOS equity curve (chained daily returns across all folds)
    concat_df = pd.DataFrame(concatenated_daily_returns, columns=["date", "daily_return"])
    concat_df["cum_equity"] = 100000.0 * (1.0 + concat_df["daily_return"]).cumprod()
    concat_df.to_csv(out_dir / "walkforward_concatenated_daily_returns.csv", index=False)

    rets = concat_df["daily_return"].to_numpy()
    total_return_pct = (concat_df["cum_equity"].iloc[-1] / 100000.0 - 1.0) * 100 if len(concat_df) else 0.0
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)) if len(rets) > 1 else 0.0
    equity_vals = concat_df["cum_equity"].to_numpy()
    peak = np.maximum.accumulate(equity_vals) if len(equity_vals) else np.array([100000.0])
    max_dd_pct = float(np.max((peak - equity_vals) / peak) * 100) if len(equity_vals) else 0.0

    summary = {
        "protocol_doc": "docs/PREREGISTERED_STRATEGY_PROTOCOL.md",
        "grid_size_per_fold": len(grid),
        "folds": fold_summaries,
        "concatenated_oos": {
            "start": concat_df["date"].iloc[0] if len(concat_df) else None,
            "end": concat_df["date"].iloc[-1] if len(concat_df) else None,
            "n_trading_days": len(concat_df),
            "total_return_pct": float(total_return_pct),
            "annualized_sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
        },
    }
    (out_dir / "walkforward_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== Walk-forward summary ===")
    print(json.dumps(summary["concatenated_oos"], indent=2))


if __name__ == "__main__":
    main()
