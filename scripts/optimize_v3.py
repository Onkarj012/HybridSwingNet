#!/usr/bin/env python3
"""Strategy Optimizer v3 — Dynamic rolling stock selection (no look-ahead)."""

import sys, logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd, numpy as np
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from backtest_historical import (
    HistoricalDataLoader, Prediction, ProfileConfig, PortfolioBacktester,
)
from stockxpert.config import load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("OptV3")

DATA_DIR = "data/yfinance_daily"
CSV_2025 = "runs/historical_backtest_20260505_153059/hit_rate_details.csv"
CSV_2026 = "runs/historical_backtest_20260505_153058/hit_rate_details.csv"
START_2025, END_2025 = "2025-01-01", "2025-12-31"
START_2026, END_2026 = "2026-01-01", "2026-04-30"
ROLLING_WINDOW = 63  # trading days (~3 months)


def load_preds(csv_path):
    """Load predictions and fill actuals from CSV."""
    df = pd.read_csv(csv_path)
    preds = []
    for _, row in df.iterrows():
        preds.append(Prediction(
            date=row['date'], symbol=row['symbol'],
            horizon=int(row['horizon']), p_up=float(row['p_up']),
            predicted_direction=row['predicted_dir'],
            predicted_magnitude=float(row['pred_magnitude']),
            confidence=float(row['confidence']),
            current_price=float(row['current_price']),
            vol_ref=0.015, atr=float(row['current_price']) * 0.02,
        ))
    adf = pd.read_csv(csv_path)
    amap = {}
    for _, row in adf.iterrows():
        amap[(row['date'], row['symbol'], int(row['horizon']))] = (
            float(row['actual_return_pct']), row['actual_dir'],
            str(row['is_correct']).lower() == 'true'
        )
    for p in preds:
        key = (p.date, p.symbol, p.horizon)
        if key in amap:
            p.actual_return_pct, p.actual_direction, p.is_correct = amap[key]
    return preds


def build_rolling_ranks(preds, horizon: int) -> Dict[str, pd.Series]:
    """Build per-symbol rolling accuracy series for dynamic stock selection.
    
    Returns: {symbol: Series(date -> rolling_accuracy)}
    Uses rolling 63-day window with 21-day minimum — NO look-ahead.
    """
    hdf = pd.DataFrame([{
        'date': pd.Timestamp(p.date), 'symbol': p.symbol,
        'correct': 1 if p.is_correct else 0,
    } for p in preds if p.horizon == horizon])
    
    hdf = hdf.sort_values(['symbol', 'date'])
    
    ranks = {}
    for sym, grp in hdf.groupby('symbol'):
        grp = grp.set_index('date').sort_index()
        rolling = grp['correct'].rolling(ROLLING_WINDOW, min_periods=21).mean()
        ranks[sym] = rolling
    return ranks


def get_top_stocks(rolling_ranks: Dict[str, pd.Series], as_of_date: str, n: int) -> List[str]:
    """Get top-N stocks by rolling accuracy as of a given date (no look-ahead).
    
    Uses accuracy data UP TO BUT NOT INCLUDING as_of_date.
    Returns empty list if insufficient history.
    """
    dt = pd.Timestamp(as_of_date)
    scores = {}
    for sym, series in rolling_ranks.items():
        # Only use data BEFORE as_of_date (exclusive)
        pre = series[series.index < dt]
        if len(pre) > 0 and not pd.isna(pre.iloc[-1]):
            scores[sym] = pre.iloc[-1]
    
    if not scores:
        return []
    # Return top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:n]]


def run_backtest(preds, daily_data, loader, profile, trading_dates,
                 stock_n: int, horizon_for_ranking: int, regime_filter: bool):
    """Run backtest with DYNAMIC stock selection — re-evaluated monthly."""
    
    # Build rolling accuracy ranks
    rolling = build_rolling_ranks(preds, horizon_for_ranking)
    
    # Pre-build prediction index by date
    by_date: Dict[str, List[Prediction]] = defaultdict(list)
    for p in preds:
        by_date[p.date].append(p)

    # Pre-compute market regime
    regime_bullish = {}
    if regime_filter:
        for td in trading_dates:
            ds = td.strftime("%Y-%m-%d")
            rets = []
            for sym in list(daily_data.keys()):
                df = daily_data.get(sym)
                if df is None: continue
                mask = df.index.strftime("%Y-%m-%d") == ds
                if not mask.any(): continue
                idx = df.index.get_loc(df.index[mask][-1])
                if idx >= 21:
                    r21 = df.iloc[idx]['Close'] / df.iloc[idx - 21]['Close'] - 1
                    rets.append(r21)
            regime_bullish[ds] = np.median(rets) > 0 if rets else True

    bt = PortfolioBacktester(profile, loader)
    current_top = []
    last_rebalance_month = ""

    for i, td in enumerate(trading_dates):
        ds = td.strftime("%Y-%m-%d")
        month_key = ds[:7]  # YYYY-MM

        # Rebalance stock selection monthly (no look-ahead)
        if month_key != last_rebalance_month:
            current_top = get_top_stocks(rolling, ds, stock_n)
            last_rebalance_month = month_key

        # Filter predictions to current top stocks only
        dps = [p for p in by_date.get(ds, []) if p.symbol in current_top]

        nd = trading_dates[i + 1].strftime("%Y-%m-%d") if i + 1 < len(trading_dates) else None

        # Regime filter: skip entries in bear markets
        if regime_filter and not regime_bullish.get(ds, True):
            bt._evaluate_exits(ds, daily_data)
            bt._check_horizon_expiry(ds, daily_data)
            pv = bt._get_portfolio_value(ds, daily_data)
            if bt.daily_equity:
                bt.daily_equity.append((ds, pv))
            continue

        bt.process_day(ds, dps, daily_data, nd)

    # Force-close open trades
    if bt.open_trades:
        ld = trading_dates[-1].strftime("%Y-%m-%d")
        for t in list(bt.open_trades):
            cp = loader.get_eod_price(t.symbol, ld)
            if cp:
                bt._close_trade(t, cp, "Backtest End", ld)
        bt.open_trades = []
    return bt.get_results()


def run_period(label, csv_path, start, end):
    print(f"\n{'='*95}")
    print(f"  {label} — Dynamic Rolling Stock Selection (63-day window)")
    print(f"{'='*95}")

    loader = HistoricalDataLoader(DATA_DIR)
    preds = load_preds(csv_path)

    run_cfg = load_config(f"{project_root}/run/20260302_182033_nifty100_improved_v2/config.yaml")
    symbols = loader.get_available_model_symbols(list(run_cfg.data.symbols))

    warmup = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    daily_data = loader.get_daily_data_batch(symbols, start=warmup, end=data_end)
    trading_dates = loader.get_trading_dates(symbols[0], start=start, end=end)

    # Show what stocks are initially selected (first trading month)
    rolling = build_rolling_ranks(preds, horizon=1)
    first_month = trading_dates[0].strftime("%Y-%m-%d")
    top = get_top_stocks(rolling, first_month, 15)
    print(f"  Initial month top-15 (as of {first_month[:7]}): {', '.join(s.replace('.NS','') for s in top[:8])}...")
    print(f"  Rolling window: {ROLLING_WINDOW} days, min 21 days warmup")
    print()

    WS = {"stop_atr_mult": 2.0, "target_atr_mult": 3.0}

    configs = [
        # ═══ Conservative (₹1L, 3 pos, H7/H10, 4 stock-rank sizes, 0.70 conf) ═══
        ("Cons | N=15 | H7/H10 | 0.70 | NoReg", ProfileConfig(
            name="CT15", capital=100000, max_positions=3,
            min_p_up=0.70, max_p_up=0.30, horizons=[7, 10], **WS,
        ), 15, False),
        ("Cons | N=20 | H7/H10 | 0.70 | NoReg", ProfileConfig(
            name="CT20", capital=100000, max_positions=3,
            min_p_up=0.70, max_p_up=0.30, horizons=[7, 10], **WS,
        ), 20, False),
        ("Cons | N=84 | H7/H10 | 0.70 | NoReg", ProfileConfig(  # all stocks baseline
            name="CT84", capital=100000, max_positions=3,
            min_p_up=0.70, max_p_up=0.30, horizons=[7, 10], **WS,
        ), 84, False),

        # ═══ Balanced (₹5L, 5 pos, H5/H7) ═══
        ("Bal  | N=10 | H5/H7 | 0.65 | NoReg", ProfileConfig(
            name="BT10", capital=500000, max_positions=5,
            min_p_up=0.65, max_p_up=0.35, horizons=[5, 7], **WS,
        ), 10, False),
        ("Bal  | N=20 | H5/H7 | 0.65 | NoReg", ProfileConfig(
            name="BT20", capital=500000, max_positions=5,
            min_p_up=0.65, max_p_up=0.35, horizons=[5, 7], **WS,
        ), 20, False),
        ("Bal  | N=10 | H5/H7 | 0.65 | REGIME", ProfileConfig(
            name="BT10R", capital=500000, max_positions=5,
            min_p_up=0.65, max_p_up=0.35, horizons=[5, 7], **WS,
        ), 10, True),
        ("Bal  | N=20 | H5/H7 | 0.65 | REGIME", ProfileConfig(
            name="BT20R", capital=500000, max_positions=5,
            min_p_up=0.65, max_p_up=0.35, horizons=[5, 7], **WS,
        ), 20, True),

        # ═══ Aggressive (₹10L, 8 pos, H3/H5) ═══
        ("Agg  | N=15 | H3/H5 | 0.60 | NoReg", ProfileConfig(
            name="AT15", capital=1000000, max_positions=8,
            min_p_up=0.60, max_p_up=0.40, horizons=[3, 5], **WS,
        ), 15, False),
        ("Agg  | N=30 | H3/H5 | 0.60 | NoReg", ProfileConfig(
            name="AT30", capital=1000000, max_positions=8,
            min_p_up=0.60, max_p_up=0.40, horizons=[3, 5], **WS,
        ), 30, False),
        ("Agg  | N=15 | H3/H5 | 0.60 | REGIME", ProfileConfig(
            name="AT15R", capital=1000000, max_positions=8,
            min_p_up=0.60, max_p_up=0.40, horizons=[3, 5], **WS,
        ), 15, True),
        ("Agg  | N=30 | H3/H5 | 0.60 | REGIME", ProfileConfig(
            name="AT30R", capital=1000000, max_positions=8,
            min_p_up=0.60, max_p_up=0.40, horizons=[3, 5], **WS,
        ), 30, True),
    ]

    print(f"{'Strategy':<43s} {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>6s} {'Trades':>6s} {'Win%':>6s} {'PF':>5s}")
    print("-" * 85)

    results = []
    for label, profile, n_stocks, regime in configs:
        r = run_backtest(preds, daily_data, loader, profile, trading_dates,
                         stock_n=n_stocks, horizon_for_ranking=1, regime_filter=regime)
        results.append((label, r))
        print(f"{label:<43s} {r['total_return_pct']:>7.2f}% {r['sharpe_ratio']:>6.2f} {r['max_drawdown_pct']:>5.1f}% {r['total_trades']:>6d} {r['win_rate']:>5.1f}% {r['profit_factor']:>4.2f}")

    best_r = max(results, key=lambda x: x[1]['total_return_pct'])
    best_s = max(results, key=lambda x: x[1]['sharpe_ratio'])
    best_w = max(results, key=lambda x: x[1]['win_rate'])
    print(f"\n  > Best Return:  {best_r[0]} | {best_r[1]['total_return_pct']:.2f}%")
    print(f"  > Best Sharpe:  {best_s[0]} | {best_s[1]['sharpe_ratio']:.2f}")
    print(f"  > Best WinRate: {best_w[0]} | {best_w[1]['win_rate']:.2f}%")
    return results


r2025 = run_period("2025 FULL YEAR (Jan-Dec)", CSV_2025, START_2025, END_2025)
r2026 = run_period("2026 JAN-APRIL", CSV_2026, START_2026, END_2026)

print(f"\n{'='*95}")
print("  CROSS-PERIOD SUMMARY (dynamic rolling stock selection)")
print(f"{'='*95}")

combined = defaultdict(list)
for label, r in r2025 + r2026:
    risk = label.split("|")[0].strip()
    combined[risk].append(r)

for risk, results in [("Cons", combined["Cons"]), ("Bal", combined["Bal"]), ("Agg", combined["Agg"])]:
    if results:
        avg_ret = np.mean([r['total_return_pct'] for r in results])
        avg_sh = np.mean([r['sharpe_ratio'] for r in results])
        best = max(results, key=lambda r: r['total_return_pct'])
        print(f"  {risk:<15s} avg_ret={avg_ret:+.1f}%  avg_sharpe={avg_sh:.2f}  best={best['total_return_pct']:+.1f}% ({best['profile']})")
print()
