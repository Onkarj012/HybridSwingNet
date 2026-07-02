"""
Intraday feature engineering from minute-bar data (VECTORIZED).

Computes daily aggregated features from minute-bar data that capture
institutional activity and market microstructure patterns invisible
in daily OHLCV data.

This version uses vectorized pandas groupby operations instead of
per-date Python loops for 100x speedup on large datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger("stockxpert.features.intraday")


def compute_intraday_features(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute intraday features for all dates using vectorized groupby.

    Args:
        daily_df: Daily OHLCV DataFrame with DateTimeIndex
        minute_df: Full minute-bar DataFrame with DateTimeIndex

    Returns:
        DataFrame with intraday feature columns, same index as daily_df
    """
    if minute_df.empty or len(minute_df) < 10:
        return _empty_features(daily_df)

    # Add a date column for grouping
    mdf = minute_df.copy()
    mdf["_date"] = mdf.index.date

    # ── Vectorized daily aggregation ──
    grouped = mdf.groupby("_date")

    # Basic OHLCV per day
    day_open = grouped["Open"].first()
    day_close = grouped["Close"].last()
    day_high = grouped["High"].max()
    day_low = grouped["Low"].min()
    day_volume = grouped["Volume"].sum().astype(float).replace(0, 1.0)
    bar_count = grouped.size()

    day_range = (day_high - day_low).replace(0, 1e-10)

    # ── 1. First-hour features ──
    def first_n_bars(g, n=60):
        return g.head(n)

    def last_n_bars(g, n=60):
        return g.tail(n)

    first_hour = grouped.apply(lambda g: g.head(min(60, len(g))), include_groups=False)
    last_hour = grouped.apply(lambda g: g.tail(min(60, len(g))), include_groups=False)

    # First hour close and volume
    fh_close = first_hour.groupby(level=0)["Close"].last()
    fh_volume = first_hour.groupby(level=0)["Volume"].sum().astype(float)

    # Last hour open and volume
    lh_open = last_hour.groupby(level=0)["Open"].first()
    lh_volume = last_hour.groupby(level=0)["Volume"].sum().astype(float)

    first_hour_return = (fh_close - day_open) / day_open
    last_hour_return = (day_close - lh_open) / lh_open

    # Volume concentration
    vol_first_pct = fh_volume / day_volume
    vol_last_pct = lh_volume / day_volume

    # ── 2. VWAP ──
    mdf["_tp"] = (mdf["High"] + mdf["Low"] + mdf["Close"]) / 3.0
    mdf["_tpv"] = mdf["_tp"] * mdf["Volume"]
    vwap = mdf.groupby("_date")["_tpv"].sum() / day_volume
    vwap_distance = (day_close - vwap) / vwap

    # ── 3. Close location ──
    close_location = (day_close - day_low) / day_range

    # ── 4. Morning reversal ──
    fh_dir = np.sign(fh_close - day_open)
    full_dir = np.sign(day_close - day_open)
    morning_reversal = (fh_dir != full_dir).astype(float)
    morning_reversal[fh_dir == 0] = 0.0

    # ── 5. Price acceleration ──
    mid_close = grouped["Close"].apply(lambda g: g.iloc[len(g)//2] if len(g) > 1 else g.iloc[0])
    first_half_ret = (mid_close - day_open) / day_open
    second_half_ret = (day_close - mid_close) / mid_close
    price_acceleration = second_half_ret - first_half_ret

    # ── 6. Volume momentum (up-vol vs down-vol) ──
    mdf["_close_diff"] = mdf.groupby("_date")["Close"].diff()
    mdf["_up_vol"] = mdf["Volume"].where(mdf["_close_diff"] > 0, 0)
    mdf["_dn_vol"] = mdf["Volume"].where(mdf["_close_diff"] < 0, 0)
    up_vol = mdf.groupby("_date")["_up_vol"].sum().astype(float)
    dn_vol = mdf.groupby("_date")["_dn_vol"].sum().astype(float)
    volume_momentum = (up_vol - dn_vol) / day_volume

    # ── 7. High/Low timing ──
    high_timing = grouped["High"].apply(lambda g: g.values.argmax() / max(len(g), 1))
    low_timing = grouped["Low"].apply(lambda g: g.values.argmin() / max(len(g), 1))

    # ── 8. Intraday volatility ratio ──
    def calc_vol_ratio(g):
        if len(g) < 5:
            return 1.0
        log_rets = np.diff(np.log(g["Close"].values + 1e-10))
        minute_vol = np.std(log_rets) * np.sqrt(len(log_rets))
        ratio = g["Close"].values[-1] / (g["Open"].values[0] + 1e-10)
        daily_ret_abs = abs(np.log(max(ratio, 1e-10)))
        return min(minute_vol / (daily_ret_abs + 1e-10), 50.0)

    intraday_vol_ratio = grouped.apply(calc_vol_ratio, include_groups=False)

    # ── Assemble into DataFrame ──
    features_df = pd.DataFrame({
        "first_hour_return": first_hour_return,
        "last_hour_return": last_hour_return,
        "vwap_distance": vwap_distance,
        "volume_first_hour_pct": vol_first_pct,
        "volume_last_hour_pct": vol_last_pct,
        "close_location_value": close_location,
        "intraday_vol_ratio": intraday_vol_ratio,
        "morning_reversal": morning_reversal,
        "price_acceleration": price_acceleration,
        "volume_momentum": volume_momentum,
        "high_timing": high_timing,
        "low_timing": low_timing,
    })

    # Convert index to Timestamp for alignment
    features_df.index = pd.to_datetime(features_df.index)

    # Align with daily_df index
    result = features_df.reindex(daily_df.index).fillna(0.0)

    # Clip extremes
    for col in result.columns:
        result[col] = result[col].clip(-10, 10)

    # Rolling aggregates for swing context
    result["vwap_distance_3d"] = result["vwap_distance"].rolling(3, min_periods=1).mean()
    result["volume_momentum_5d"] = result["volume_momentum"].rolling(5, min_periods=1).mean()
    result["first_hour_trend"] = result["first_hour_return"].rolling(5, min_periods=1).mean()
    result["reversal_frequency_5d"] = result["morning_reversal"].rolling(5, min_periods=1).mean()

    logger.info(f"Computed {len(result.columns)} intraday features for {len(result)} days")
    return result


def _empty_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of zeros with the correct feature columns."""
    cols = [
        "first_hour_return", "last_hour_return", "vwap_distance",
        "volume_first_hour_pct", "volume_last_hour_pct",
        "close_location_value", "intraday_vol_ratio",
        "morning_reversal", "price_acceleration",
        "volume_momentum", "high_timing", "low_timing",
        "vwap_distance_3d", "volume_momentum_5d",
        "first_hour_trend", "reversal_frequency_5d",
    ]
    return pd.DataFrame(0.0, index=daily_df.index, columns=cols)


def add_intraday_features_batch(
    daily_data: Dict[str, pd.DataFrame],
    minute_loader,
    symbols: list,
) -> Dict[str, pd.DataFrame]:
    """
    Add intraday features to all stock DataFrames.

    Args:
        daily_data: Dict of symbol -> daily DataFrame
        minute_loader: HistoricalDataLoader instance with minute data
        symbols: List of symbols to process

    Returns:
        Updated daily_data dict with intraday columns added
    """
    success = 0
    total = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        if symbol not in daily_data:
            continue

        try:
            minute_df = minute_loader._load_minute_data(symbol)
            intraday_feats = compute_intraday_features(daily_data[symbol], minute_df)

            for col in intraday_feats.columns:
                daily_data[symbol][col] = intraday_feats[col]

            success += 1
            if success % 10 == 0 or success == total:
                logger.info(f"  Intraday features: {success}/{total} stocks done")
        except (FileNotFoundError, KeyError) as e:
            logger.debug(f"No minute data for {symbol}: {e}")
            empty = _empty_features(daily_data[symbol])
            for col in empty.columns:
                daily_data[symbol][col] = 0.0

    logger.info(f"Added intraday features: {success}/{total} symbols")
    return daily_data
