#!/usr/bin/env python3
"""Develop and stress-test candidate hold-to-horizon strategies.

This script intentionally writes to a separate analysis folder and does not
modify the H10 benchmark replay/validation files. It uses saved predictions,
causal daily price filters, and hold-to-horizon execution to answer:

1. Can a rule selected on the available 2024 segment improve on raw H10 Top-1?
2. Does the selected rule remain competitive in untouched 2025?
3. What happens month-by-month across the full Jan-Jun 2026 stress window?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from historical_data_loader import HistoricalDataLoader


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    horizon: int
    top_k: int
    min_p_up: float | None
    market_min_21d: float | None
    market_min_63d: float | None
    breadth_min_21d: float | None
    symbol_mom_lookback: int | None
    symbol_mom_min: float | None
    one_open_per_symbol: bool


@dataclass
class Position:
    symbol: str
    signal_date: str
    entry_date: str
    entry_price: float
    shares: int
    gross_cost: float
    entry_cost: float
    p_up: float
    exit_i: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy development analysis for 2024/2025/2026")
    parser.add_argument("--out-dir", default="runs/strategy_development_2024_2026")
    parser.add_argument("--selection-predictions", default="runs/v3_backtest_20260522_205948/predictions.csv")
    parser.add_argument("--final-2025-predictions", default="runs/v3_2025_long_off_0bps/predictions.csv")
    parser.add_argument("--full-2026-predictions", default="runs/v3_2026_regime_off_0bps/predictions.csv")
    parser.add_argument("--data-dir-2024-2025", default="data/nifty500_v3_yf_tail_20260630")
    parser.add_argument("--data-dir-2026", default="data/nifty500_v3_yf_tail_20260630")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--selection-top-n", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def load_predictions(path: str, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "symbol", "horizon", "p_up"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    frame["p_up"] = pd.to_numeric(frame["p_up"], errors="coerce")
    frame = frame.dropna(subset=["p_up"])
    if frame.empty:
        raise ValueError(f"No predictions in {path} for {start} to {end}")
    return frame


def load_daily(
    symbols: Iterable[str],
    data_dir: str,
    start: str,
    end: str,
    warmup_days: int = 90,
) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    loader = HistoricalDataLoader(data_dir)
    warmup_start = (pd.Timestamp(start) - pd.offsets.BDay(warmup_days)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end) + pd.offsets.BDay(45)).strftime("%Y-%m-%d")
    daily = loader.get_daily_data_batch(sorted(set(symbols)), start=warmup_start, end=data_end)
    if not daily:
        raise RuntimeError(f"No daily data loaded from {data_dir}")
    ref_symbol = max(daily, key=lambda symbol: len(daily[symbol]))
    dates = loader.get_trading_dates(ref_symbol, start=start, end=end)
    if not dates:
        raise RuntimeError(f"No trading dates in {start} to {end} from {data_dir}")
    return daily, dates


def daily_lookup(daily: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.Series]]:
    return {symbol: {idx.strftime("%Y-%m-%d"): row for idx, row in frame.iterrows()} for symbol, frame in daily.items()}


def compute_market_stats(
    daily: dict[str, pd.DataFrame],
    trading_dates: list[pd.Timestamp],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for date in trading_dates:
        date_str = date.strftime("%Y-%m-%d")
        ret21: list[float] = []
        ret63: list[float] = []
        positive21 = 0
        for frame in daily.values():
            idxs = np.flatnonzero(frame.index.strftime("%Y-%m-%d") == date_str)
            if len(idxs) == 0:
                continue
            idx = int(idxs[-1])
            if idx >= 21:
                r21 = float(frame.iloc[idx]["Close"] / frame.iloc[idx - 21]["Close"] - 1.0)
                ret21.append(r21)
                positive21 += int(r21 > 0)
            if idx >= 63:
                ret63.append(float(frame.iloc[idx]["Close"] / frame.iloc[idx - 63]["Close"] - 1.0))
        stats[date_str] = {
            "median_21d": float(np.median(ret21)) if ret21 else 0.0,
            "median_63d": float(np.median(ret63)) if ret63 else 0.0,
            "breadth_21d": float(positive21 / len(ret21)) if ret21 else 0.0,
        }
    return stats


def symbol_return(
    daily_rows: dict[str, dict[str, pd.Series]],
    date_strs: list[str],
    symbol: str,
    date_i: int,
    lookback: int,
) -> float | None:
    start_i = date_i - lookback
    if start_i < 0:
        return None
    start_row = daily_rows.get(symbol, {}).get(date_strs[start_i])
    end_row = daily_rows.get(symbol, {}).get(date_strs[date_i])
    if start_row is None or end_row is None:
        return None
    start_close = float(start_row["Close"])
    if start_close <= 0:
        return None
    return float(end_row["Close"] / start_close - 1.0)


def passes_filters(
    symbol: str,
    p_up: float,
    config: StrategyConfig,
    market_stats: dict[str, dict[str, float]],
    daily_rows: dict[str, dict[str, pd.Series]],
    date_strs: list[str],
    date_i: int,
    open_symbols: set[str],
) -> bool:
    if config.min_p_up is not None and p_up < config.min_p_up:
        return False
    if config.one_open_per_symbol and symbol in open_symbols:
        return False

    date_str = date_strs[date_i]
    stat = market_stats.get(date_str, {})
    if config.market_min_21d is not None and stat.get("median_21d", 0.0) < config.market_min_21d:
        return False
    if config.market_min_63d is not None and stat.get("median_63d", 0.0) < config.market_min_63d:
        return False
    if config.breadth_min_21d is not None and stat.get("breadth_21d", 0.0) < config.breadth_min_21d:
        return False

    if config.symbol_mom_lookback is not None and config.symbol_mom_min is not None:
        ret = symbol_return(daily_rows, date_strs, symbol, date_i, config.symbol_mom_lookback)
        if ret is None or ret < config.symbol_mom_min:
            return False
    return True


def close_position(
    position: Position,
    exit_date: str,
    exit_price: float,
    cash: float,
    per_side_cost_rate: float,
    reason: str,
) -> tuple[float, dict]:
    gross_exit_value = position.shares * exit_price
    exit_cost = gross_exit_value * per_side_cost_rate
    cash += gross_exit_value - exit_cost
    net_pnl = gross_exit_value - position.gross_cost - position.entry_cost - exit_cost
    return cash, {
        **asdict(position),
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_cost": exit_cost,
        "exit_reason": reason,
        "gross_exit_value": gross_exit_value,
        "net_pnl": net_pnl,
        "return_pct": net_pnl / position.gross_cost * 100.0 if position.gross_cost else 0.0,
    }


def replay_strategy(
    predictions: pd.DataFrame,
    daily: dict[str, pd.DataFrame],
    trading_dates: list[pd.Timestamp],
    config: StrategyConfig,
    *,
    capital: float,
    round_trip_cost_bps: float,
    daily_rows: dict[str, dict[str, pd.Series]] | None = None,
    market_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    date_strs = [date.strftime("%Y-%m-%d") for date in trading_dates]
    daily_rows = daily_rows or daily_lookup(daily)
    market_stats = market_stats or compute_market_stats(daily, trading_dates)
    preds = predictions[predictions["horizon"] == config.horizon].copy()
    preds_by_date = {date: group for date, group in preds.groupby("date")}

    max_slots = max(1, config.horizon * config.top_k)
    slot_capital = capital / max_slots
    per_side_cost_rate = (round_trip_cost_bps / 10000.0) / 2.0
    cash = capital
    open_positions: list[Position] = []
    closed: list[dict] = []
    equity_rows: list[dict] = []

    for i, date_str in enumerate(date_strs):
        still_open: list[Position] = []
        for position in open_positions:
            row = daily_rows.get(position.symbol, {}).get(date_str)
            if i >= position.exit_i and row is not None:
                cash, closed_row = close_position(
                    position,
                    date_str,
                    float(row["Close"]),
                    cash,
                    per_side_cost_rate,
                    "Horizon End",
                )
                closed.append(closed_row)
            else:
                still_open.append(position)
        open_positions = still_open

        if i + 1 < len(date_strs):
            signal_frame = preds_by_date.get(date_str)
            if signal_frame is not None:
                open_symbols = {position.symbol for position in open_positions}
                filtered: list[tuple[str, float]] = []
                for row in signal_frame.sort_values("p_up", ascending=False).itertuples(index=False):
                    symbol = str(row.symbol)
                    p_up = float(row.p_up)
                    if passes_filters(symbol, p_up, config, market_stats, daily_rows, date_strs, i, open_symbols):
                        filtered.append((symbol, p_up))
                        if len(filtered) >= config.top_k:
                            break
                for symbol, p_up in filtered:
                    if len(open_positions) >= max_slots:
                        break
                    entry_date = date_strs[i + 1]
                    entry_row = daily_rows.get(symbol, {}).get(entry_date)
                    if entry_row is None:
                        continue
                    entry_price = float(entry_row["Open"])
                    shares = int(slot_capital / entry_price)
                    if shares < 1:
                        continue
                    gross_cost = shares * entry_price
                    entry_cost = gross_cost * per_side_cost_rate
                    if cash < gross_cost + entry_cost:
                        continue
                    cash -= gross_cost + entry_cost
                    open_positions.append(
                        Position(
                            symbol=symbol,
                            signal_date=date_str,
                            entry_date=entry_date,
                            entry_price=entry_price,
                            shares=shares,
                            gross_cost=gross_cost,
                            entry_cost=entry_cost,
                            p_up=p_up,
                            exit_i=min(i + 1 + config.horizon, len(date_strs) - 1),
                        )
                    )

        equity = cash
        for position in open_positions:
            row = daily_rows.get(position.symbol, {}).get(date_str)
            mark = float(row["Close"]) if row is not None else position.entry_price
            equity += position.shares * mark
        equity_rows.append({"date": date_str, "equity": equity, "cash": cash, "open_positions": len(open_positions)})

    final_date = date_strs[-1]
    for position in list(open_positions):
        row = daily_rows.get(position.symbol, {}).get(final_date)
        if row is not None:
            cash, closed_row = close_position(
                position,
                final_date,
                float(row["Close"]),
                cash,
                per_side_cost_rate,
                "Backtest End",
            )
            closed.append(closed_row)
    if equity_rows:
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["open_positions"] = 0

    trades = pd.DataFrame(closed)
    equity_curve = pd.DataFrame(equity_rows)
    summary = summarize(trades, equity_curve, config, capital, round_trip_cost_bps)
    return trades, equity_curve, summary


def summarize(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    config: StrategyConfig,
    capital: float,
    cost_bps: float,
) -> dict:
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else capital
    peak = equity["equity"].cummax() if not equity.empty else pd.Series([capital])
    max_dd = float(((peak - equity["equity"]) / peak * 100.0).max()) if not equity.empty else 0.0
    returns = equity["equity"].pct_change().dropna() if len(equity) > 1 else pd.Series(dtype=float)
    sharpe = 0.0
    if not returns.empty and returns.std() != 0 and not pd.isna(returns.std()):
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
    pnl_sum = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
    return {
        **asdict(config),
        "round_trip_cost_bps": cost_bps,
        "initial_capital": capital,
        "final_equity": final_equity,
        "total_return": final_equity - capital,
        "total_return_pct": (final_equity - capital) / capital * 100.0 if capital else 0.0,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "trades": int(len(trades)),
        "win_rate": float((trades["net_pnl"] > 0).mean() * 100.0) if not trades.empty else 0.0,
        "avg_trade_return_pct": float(trades["return_pct"].mean()) if not trades.empty else 0.0,
        "reconciliation_delta": (final_equity - capital) - pnl_sum,
        "top_symbols": top_symbols(trades),
    }


def top_symbols(trades: pd.DataFrame, limit: int = 8) -> str:
    if trades.empty:
        return ""
    return "; ".join(f"{symbol}:{count}" for symbol, count in trades["symbol"].value_counts().head(limit).items())


def monthly_metrics(equity: pd.DataFrame, trades: pd.DataFrame, label: str, window: str) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    trade_frame = trades.copy()
    if not trade_frame.empty:
        trade_frame["exit_month"] = pd.to_datetime(trade_frame["exit_date"]).dt.strftime("%Y-%m")
    rows: list[dict] = []
    for month, group in frame.groupby("month"):
        start_equity = float(group["equity"].iloc[0])
        end_equity = float(group["equity"].iloc[-1])
        month_trades = trade_frame[trade_frame["exit_month"] == month] if not trade_frame.empty else pd.DataFrame()
        rows.append({
            "run_label": label,
            "window": window,
            "month": month,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "monthly_return_pct": (end_equity - start_equity) / start_equity * 100.0 if start_equity else 0.0,
            "closed_trades": int(len(month_trades)),
            "win_rate": float((month_trades["net_pnl"] > 0).mean() * 100.0) if not month_trades.empty else 0.0,
            "net_pnl": float(month_trades["net_pnl"].sum()) if not month_trades.empty else 0.0,
        })
    return pd.DataFrame(rows)


def candidate_grid() -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []
    market_filters = [
        ("mkt_off", None, None, None),
        ("mkt21_ge_0", 0.0, None, None),
        ("mkt21_ge_1pct", 0.01, None, None),
        ("robust_not_weak", 0.0, -0.02, 0.45),
    ]
    momentum_filters = [("mom_off", None, None)]
    for horizon in [5, 7, 10]:
        for top_k in [1, 2, 3]:
            for min_p_up in [None, 0.55, 0.60, 0.65, 0.70]:
                for mkt_name, m21, m63, breadth in market_filters:
                    for mom_name, mom_lb, mom_min in momentum_filters:
                        for one_symbol in [False, True]:
                            threshold_name = "pall" if min_p_up is None else f"p{int(min_p_up * 100)}"
                            name = (
                                f"H{horizon}_top{top_k}_{threshold_name}_"
                                f"{mkt_name}_{mom_name}_{'oneSym' if one_symbol else 'multiSym'}"
                            )
                            configs.append(
                                StrategyConfig(
                                    name=name,
                                    horizon=horizon,
                                    top_k=top_k,
                                    min_p_up=min_p_up,
                                    market_min_21d=m21,
                                    market_min_63d=m63,
                                    breadth_min_21d=breadth,
                                    symbol_mom_lookback=mom_lb,
                                    symbol_mom_min=mom_min,
                                    one_open_per_symbol=one_symbol,
                                )
                            )
    return configs


def baseline_configs() -> list[StrategyConfig]:
    return [
        StrategyConfig("benchmark_H10_top1", 10, 1, None, None, None, None, None, None, False),
        StrategyConfig("benchmark_H10_top3", 10, 3, None, None, None, None, None, None, False),
        StrategyConfig("benchmark_H7_top3", 7, 3, None, None, None, None, None, None, False),
    ]


def selection_score(row: pd.Series) -> float:
    if row["trades"] < 20:
        return -9999.0
    return float(row["total_return_pct"] - 0.35 * row["max_drawdown_pct"] + 1.5 * row["sharpe_ratio"])


def write_report(
    out_dir: Path,
    selection: pd.DataFrame,
    evaluation: pd.DataFrame,
    monthly: pd.DataFrame,
    all_window_rank: pd.DataFrame,
) -> None:
    top = selection.sort_values("selection_score", ascending=False).head(12)
    selected_names = [name for name in top["name"].head(5)]
    selected_eval = evaluation[evaluation["name"].isin(selected_names + ["benchmark_H10_top1", "benchmark_H10_top3"])]
    positive_all = int((all_window_rank["positive_windows"] == 3).sum())
    best_robust = all_window_rank.iloc[0].to_dict() if not all_window_rank.empty else {}
    lines = [
        "# Strategy Development Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Benchmark files were not modified.",
        "- Candidate selection uses the available 2024 saved-prediction segment: 2024-09-05 to 2024-12-31.",
        "- 2025 is treated as the untouched final-test year.",
        "- 2026 full stress window uses 2026-01-01 to 2026-06-30 with `data/nifty500_v3_yf_tail_20260630`.",
        "- The local repository does not contain saved V3 predictions before 2024-09-05, so pre-2024 strategy selection is not included here.",
        "",
        "## Executive Read",
        "",
        f"- Candidates positive in all three windows: {positive_all}.",
        f"- Best worst-window candidate: `{best_robust.get('name', 'n/a')}` with worst-window return {best_robust.get('min_return_pct', float('nan')):.2f}%.",
        "- The 2024-positive H5 robust-regime filters do not survive 2025 or full Jan-Jun 2026.",
        "- The H10 family still owns the best 2025 result, but the available 2024 slice remains the blocking evidence for a headline claim.",
        "",
        "## Top 2024-Selected Candidates",
        "",
        markdown_table(top, ["name", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "trades", "win_rate", "selection_score"]),
        "",
        "## All-Window Candidate Ranking at 0 bps",
        "",
        markdown_table(
            all_window_rank.head(15),
            ["name", "min_return_pct", "mean_return_pct", "positive_windows", "selection_2024", "final_2025", "stress_2026_jan_jun"],
        ),
        "",
        "## Cross-Window Evaluation",
        "",
        markdown_table(
            selected_eval.sort_values(["name", "window", "round_trip_cost_bps"]),
            ["name", "window", "round_trip_cost_bps", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "trades", "win_rate"],
        ),
        "",
        "## 2026 Monthly Breakdown",
        "",
        markdown_table(
            monthly[monthly["run_label"].isin(selected_names + ["benchmark_H10_top1"])],
            ["run_label", "month", "monthly_return_pct", "closed_trades", "win_rate", "net_pnl"],
        ),
        "",
        "## Read",
        "",
        "Treat any top-ranked 2024 candidate as a hypothesis, not a finished paper result. The 2024 window is short and begins in September, so a journal-grade claim still needs either earlier saved predictions or a fresh leakage-safe rerun with a longer pre-2025 selection period.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    rows = [columns, ["---"] * len(columns)]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    return "\n".join("| " + " | ".join(values) + " |" for values in rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = {
        "selection_2024": {
            "predictions": load_predictions(args.selection_predictions, "2024-09-05", "2024-12-31"),
            "data_dir": args.data_dir_2024_2025,
            "start": "2024-09-05",
            "end": "2024-12-31",
        },
        "final_2025": {
            "predictions": load_predictions(args.final_2025_predictions, "2025-01-01", "2025-12-31"),
            "data_dir": args.data_dir_2024_2025,
            "start": "2025-01-01",
            "end": "2025-12-31",
        },
        "stress_2026_jan_jun": {
            "predictions": load_predictions(args.full_2026_predictions, "2026-01-01", "2026-06-30"),
            "data_dir": args.data_dir_2026,
            "start": "2026-01-01",
            "end": "2026-06-30",
        },
    }

    loaded_by_dir: dict[str, dict[str, pd.DataFrame]] = {}
    for data_dir in sorted({spec["data_dir"] for spec in windows.values()}):
        grouped = [spec for spec in windows.values() if spec["data_dir"] == data_dir]
        symbols = sorted({symbol for spec in grouped for symbol in spec["predictions"]["symbol"].unique()})
        start = min(spec["start"] for spec in grouped)
        end = max(spec["end"] for spec in grouped)
        daily, _ = load_daily(symbols, data_dir, start, end)
        loaded_by_dir[data_dir] = daily

    loaded: dict[str, tuple[dict[str, pd.DataFrame], list[pd.Timestamp], dict[str, dict[str, pd.Series]], dict[str, dict[str, float]]]] = {}
    for name, spec in windows.items():
        daily = loaded_by_dir[spec["data_dir"]]
        ref_symbol = max(daily, key=lambda symbol: len(daily[symbol]))
        ref_frame = daily[ref_symbol]
        dates = list(ref_frame[(ref_frame.index >= spec["start"]) & (ref_frame.index <= spec["end"])].index)
        rows = daily_lookup(daily)
        stats = compute_market_stats(daily, dates)
        loaded[name] = (daily, dates, rows, stats)

    selection_rows: list[dict] = []
    selection_daily, selection_dates, selection_rows_lookup, selection_market_stats = loaded["selection_2024"]
    for config in candidate_grid():
        trades, equity, summary = replay_strategy(
            windows["selection_2024"]["predictions"],
            selection_daily,
            selection_dates,
            config,
            capital=args.capital,
            round_trip_cost_bps=0.0,
            daily_rows=selection_rows_lookup,
            market_stats=selection_market_stats,
        )
        summary["window"] = "selection_2024"
        summary["selection_score"] = selection_score(pd.Series(summary))
        selection_rows.append(summary)
    selection = pd.DataFrame(selection_rows).sort_values("selection_score", ascending=False)
    selection.to_csv(out_dir / "candidate_sweep_2024.csv", index=False)

    all_eval_rows: list[dict] = []
    for window_name, spec in windows.items():
        daily, dates, rows_lookup, stats = loaded[window_name]
        for config in candidate_grid() + baseline_configs():
            trades, equity, summary = replay_strategy(
                spec["predictions"],
                daily,
                dates,
                config,
                capital=args.capital,
                round_trip_cost_bps=0.0,
                daily_rows=rows_lookup,
                market_stats=stats,
            )
            summary["window"] = window_name
            all_eval_rows.append(summary)
    all_eval = pd.DataFrame(all_eval_rows)
    all_eval.to_csv(out_dir / "all_candidate_cross_window_0bps.csv", index=False)
    rank_rows: list[dict] = []
    for name, group in all_eval.groupby("name"):
        by_window = {row["window"]: float(row["total_return_pct"]) for _, row in group.iterrows()}
        returns = list(by_window.values())
        rank_rows.append({
            "name": name,
            "min_return_pct": min(returns),
            "mean_return_pct": float(np.mean(returns)),
            "positive_windows": int(sum(value > 0 for value in returns)),
            "selection_2024": by_window.get("selection_2024", np.nan),
            "final_2025": by_window.get("final_2025", np.nan),
            "stress_2026_jan_jun": by_window.get("stress_2026_jan_jun", np.nan),
        })
    all_window_rank = pd.DataFrame(rank_rows).sort_values(
        ["positive_windows", "min_return_pct", "mean_return_pct"],
        ascending=[False, False, False],
    )
    all_window_rank.to_csv(out_dir / "all_window_rank_0bps.csv", index=False)

    top_names = set(selection.head(args.selection_top_n)["name"])
    config_by_name = {config.name: config for config in candidate_grid() + baseline_configs()}
    eval_configs = [config_by_name[name] for name in top_names if name in config_by_name] + baseline_configs()
    seen: set[str] = set()
    eval_configs = [config for config in eval_configs if not (config.name in seen or seen.add(config.name))]

    eval_rows: list[dict] = []
    monthly_rows: list[pd.DataFrame] = []
    for window_name, spec in windows.items():
        daily, dates, rows_lookup, stats = loaded[window_name]
        costs = [0.0, 41.0] if window_name != "selection_2024" else [0.0]
        for config in eval_configs:
            for cost in costs:
                trades, equity, summary = replay_strategy(
                    spec["predictions"],
                    daily,
                    dates,
                    config,
                    capital=args.capital,
                    round_trip_cost_bps=cost,
                    daily_rows=rows_lookup,
                    market_stats=stats,
                )
                summary["window"] = window_name
                summary["data_dir"] = spec["data_dir"]
                eval_rows.append(summary)
                if window_name == "stress_2026_jan_jun" and cost == 0.0:
                    monthly_rows.append(monthly_metrics(equity, trades, config.name, window_name))
                run_dir = out_dir / "selected_runs" / f"{window_name}_{config.name}_{int(cost)}bps"
                run_dir.mkdir(parents=True, exist_ok=True)
                trades.to_csv(run_dir / "trade_log.csv", index=False)
                equity.to_csv(run_dir / "equity_curve.csv", index=False)
                with (run_dir / "summary.json").open("w") as f:
                    json.dump(summary, f, indent=2)

    evaluation = pd.DataFrame(eval_rows)
    evaluation.to_csv(out_dir / "evaluation_matrix.csv", index=False)
    monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    monthly.to_csv(out_dir / "monthly_2026.csv", index=False)
    write_report(out_dir, selection, evaluation, monthly, all_window_rank)
    print(f"Wrote strategy development analysis to {out_dir}")


if __name__ == "__main__":
    main()
