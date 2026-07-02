#!/usr/bin/env python3
"""Download yfinance daily data for Nifty100 symbols and save as pseudo-minute CSVs."""
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from stockxpert.config import load_config

# Load the model's config (run config for exact symbol list)
cfg = load_config(str(project_root / "configs" / "nifty100_improved_v2.yaml"))
symbols = cfg.data.symbols

# Also load run config for data range
run_cfg = load_config(str(project_root / "run" / "20260302_182033_nifty100_improved_v2" / "config.yaml"))
data_end = pd.Timestamp("2026-04-30")  # extend to cover test period

out_dir = project_root / "data" / "yfinance_daily"
out_dir.mkdir(parents=True, exist_ok=True)

# Symbol mapping (same as HistoricalDataLoader)
SYMBOL_OVERRIDES = {"M&M.NS": "MM"}

print(f"Downloading {len(symbols)} symbols from yfinance...")
print(f"Date range: 2013-01-01 to {data_end.strftime('%Y-%m-%d')}")

for i, sym in enumerate(symbols):
    basename = SYMBOL_OVERRIDES.get(sym, sym.replace(".NS", "").replace(".BO", ""))
    out_path = out_dir / f"{basename}_minute.csv"

    if out_path.exists():
        # Check if the file already has data up to our end date
        existing = pd.read_csv(out_path, parse_dates=["date"], index_col="date")
        if not existing.empty:
            last_date = existing.index[-1]
            if pd.Timestamp(last_date) >= data_end:
                print(f"  [{i+1}/{len(symbols)}] {sym} -> {basename} (already up to date)")
                continue

    try:
        df = yf.download(
            sym,
            start="2013-01-01",
            end=data_end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            print(f"  [{i+1}/{len(symbols)}] {sym} -> {basename}: NO DATA")
            continue

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # yfinance gives: Open, High, Low, Close, Volume
        # Create pseudo-minute format with 15:30 timestamps
        rows = []
        for idx, row in df.iterrows():
            ts = pd.Timestamp(f"{idx.strftime('%Y-%m-%d')} 15:30:00")
            rows.append({
                "date": ts,
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            })

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        print(f"  [{i+1}/{len(symbols)}] {sym} -> {basename} ({len(out_df)} days)")

    except Exception as e:
        print(f"  [{i+1}/{len(symbols)}] {sym} -> {basename}: ERROR {e}")

print(f"\nDone! Files saved to {out_dir}")
print(f"File count: {len(list(out_dir.glob('*_minute.csv')))}")
