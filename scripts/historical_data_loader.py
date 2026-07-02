#!/usr/bin/env python3
"""
Historical Data Loader for StockXpert Backtesting

Loads minute-bar CSV files from data/nifty500/ and resamples to daily OHLCV
for use with the StockXpert model pipeline. Also provides minute-level data
access for precise exit evaluation.

Usage:
    from historical_data_loader import HistoricalDataLoader
    loader = HistoricalDataLoader("data/nifty500")
    daily = loader.get_daily_data("INFY.NS", "2024-01-01", "2025-12-31")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("HistoricalDataLoader")


class HistoricalDataLoader:
    """
    Loads minute-bar CSVs and provides daily + minute-level access.
    
    CSV format: date,open,high,low,close,volume
    File naming: {SYMBOL}_minute.csv (e.g., INFY_minute.csv)
    """

    # Map from model symbol (e.g. "INFY.NS") to CSV base name (e.g. "INFY")
    # Special cases where the ticker doesn't directly match filename
    SYMBOL_OVERRIDES = {
        "M&M.NS": "MM",   # Mahindra & Mahindra
    }

    def __init__(self, data_dir: str = "data/nifty500"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Cache for loaded data
        self._minute_cache: Dict[str, pd.DataFrame] = {}
        self._daily_cache: Dict[str, pd.DataFrame] = {}

        # Discover available symbols
        self.available_files = {
            f.stem.replace("_minute", ""): f
            for f in self.data_dir.glob("*_minute.csv")
        }
        logger.info(f"Found {len(self.available_files)} minute-bar CSV files")

    def _symbol_to_basename(self, symbol: str) -> str:
        """Convert model symbol (INFY.NS) to CSV basename (INFY)."""
        if symbol in self.SYMBOL_OVERRIDES:
            return self.SYMBOL_OVERRIDES[symbol]
        # Strip .NS / .BO suffix
        base = symbol.replace(".NS", "").replace(".BO", "")
        return base

    def has_data(self, symbol: str) -> bool:
        """Check if data exists for a symbol."""
        base = self._symbol_to_basename(symbol)
        return base in self.available_files

    def get_available_model_symbols(self, model_symbols: List[str]) -> List[str]:
        """Filter model symbols to those with available CSV data."""
        available = []
        for sym in model_symbols:
            if self.has_data(sym):
                available.append(sym)
        logger.info(f"{len(available)}/{len(model_symbols)} model symbols have data")
        return available

    def _load_minute_data(self, symbol: str) -> pd.DataFrame:
        """Load raw minute-bar data for a symbol (cached)."""
        base = self._symbol_to_basename(symbol)
        if base in self._minute_cache:
            return self._minute_cache[base]

        if base not in self.available_files:
            raise FileNotFoundError(f"No data file for {symbol} (base={base})")

        filepath = self.available_files[base]
        logger.debug(f"Loading minute data: {filepath.name}")

        df = pd.read_csv(
            filepath,
            parse_dates=["date"],
            index_col="date",
        )

        # Capitalize columns to match model expectations
        df.columns = [c.capitalize() for c in df.columns]
        # Ensure Volume is integer
        df["Volume"] = df["Volume"].fillna(0).astype(int)

        self._minute_cache[base] = df
        return df

    def get_minute_data_for_date(
        self, symbol: str, date: str
    ) -> Optional[pd.DataFrame]:
        """Get minute-level bars for a specific trading day."""
        df = self._load_minute_data(symbol)
        day = pd.Timestamp(date).normalize()
        start = df.index.searchsorted(day, side="left")
        end = df.index.searchsorted(day + pd.Timedelta(days=1), side="left")
        day_df = df.iloc[start:end]
        return day_df if not day_df.empty else None

    def _resample_to_daily(self, symbol: str) -> pd.DataFrame:
        """Resample minute data to daily OHLCV."""
        base = self._symbol_to_basename(symbol)
        if base in self._daily_cache:
            return self._daily_cache[base]

        minute_df = self._load_minute_data(symbol)

        daily = minute_df.resample("D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])

        # Remove non-trading days (weekends, holidays with no data)
        daily = daily[daily["Volume"] > 0]

        self._daily_cache[base] = daily
        return daily

    def get_daily_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data for a symbol, optionally filtered by date range.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (date-only)
        """
        daily = self._resample_to_daily(symbol)

        if start:
            daily = daily[daily.index >= start]
        if end:
            daily = daily[daily.index <= end]

        return daily

    def get_daily_data_batch(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load daily data for multiple symbols."""
        data = {}
        failed = 0
        for sym in symbols:
            try:
                df = self.get_daily_data(sym, start=start, end=end)
                if len(df) >= 60:  # Need at least 60 days for long window
                    data[sym] = df
                else:
                    logger.warning(f"  {sym}: Only {len(df)} rows, skipping")
                    failed += 1
            except Exception as e:
                logger.warning(f"  {sym}: {e}")
                failed += 1

        logger.info(
            f"Loaded daily data: {len(data)}/{len(symbols)} symbols "
            f"({failed} skipped)"
        )
        return data

    def get_trading_dates(
        self,
        symbol: str = "INFY.NS",
        start: str = "2025-01-01",
        end: str = "2025-12-31",
    ) -> List[pd.Timestamp]:
        """Get list of trading dates from a reference stock."""
        daily = self.get_daily_data(symbol, start=start, end=end)
        return list(daily.index)

    def check_minute_exit(
        self,
        symbol: str,
        date: str,
        entry_price: float,
        target: float,
        stop_loss: float,
        side: str = "LONG",
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Check minute-by-minute if target or stop was hit on a given date.

        Returns: (exit_price, exit_reason, exit_time) or (None, None, None)
        """
        minute_df = self.get_minute_data_for_date(symbol, date)
        if minute_df is None or minute_df.empty:
            return None, None, None

        high = minute_df["High"].to_numpy(dtype=float)
        low = minute_df["Low"].to_numpy(dtype=float)

        if side == "LONG":
            stop_hits = np.flatnonzero(low <= stop_loss)
            target_hits = np.flatnonzero(high >= target)
        else:
            stop_hits = np.flatnonzero(high >= stop_loss)
            target_hits = np.flatnonzero(low <= target)

        first_stop = int(stop_hits[0]) if len(stop_hits) else None
        first_target = int(target_hits[0]) if len(target_hits) else None

        if first_stop is None and first_target is None:
            return None, None, None
        if first_target is None or (first_stop is not None and first_stop <= first_target):
            return stop_loss, "Stop Loss", str(minute_df.index[first_stop])
        return target, "Target Hit", str(minute_df.index[first_target])

        return None, None, None

    def get_eod_price(self, symbol: str, date: str) -> Optional[float]:
        """Get end-of-day close price."""
        daily = self.get_daily_data(symbol)
        day = pd.Timestamp(date).normalize()
        day_data = daily.loc[(daily.index >= day) & (daily.index < day + pd.Timedelta(days=1))]
        if day_data.empty:
            return None
        return float(day_data["Close"].iloc[0])

    def preload_symbols(self, symbols: List[str]) -> None:
        """Pre-load minute + daily data for given symbols into cache."""
        logger.info(f"Pre-loading {len(symbols)} symbols into memory...")
        for sym in symbols:
            try:
                self._resample_to_daily(sym)
            except Exception as e:
                logger.warning(f"  Failed to preload {sym}: {e}")
        logger.info(f"  Cached {len(self._daily_cache)} daily DataFrames")
