"""
Technical indicators for feature engineering.
All indicators follow consistent formulas locked in the spec.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from .fibonacci import compute_fibonacci_levels




def compute_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-asset features like VIX, Bank Nifty proxies.
    """
    # 1. Volatility Regime (VIX Proxy)
    # Annualized volatility of last 21 days
    df['vix_proxy'] = df['Close'].pct_change().rolling(21).std() * (252 ** 0.5) * 100
    
    # 2. Relative Strength vs Market (Proxy)
    # We don't have Nifty 50 loaded here, so we use self-momentum
    df['rs_proxy'] = df['Close'].pct_change(21)
    
    return df

def compute_macro_proxy(df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Computes a macro regime proxy.
    If index_df is provided (NIFTY50), uses its rolling return.
    Otherwise, uses the stock's own interaction with broad market moving averages if available.
    For now, we'll implement a simple 'Beta' proxy or just returns relative to recent history.
    """
    # Simple proxy: 21-day rolling return of the stock itself (momentum)
    # Ideally should be relative to NIFTY, but we might not have it aligned in this function scope.
    return df['Close'].pct_change(periods=21).fillna(0)

def compute_atr_change(df: pd.DataFrame, period: int = 21) -> pd.Series:
    """Percentage change in ATR over the period."""
    atr = compute_atr(df, period=14)
    return atr.pct_change(periods=period).fillna(0)


def compute_sma_cross(series: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Returns 1 if fast > slow, -1 if fast < slow.
    Captures trend direction.
    """
    sma_fast = series.rolling(window=fast).mean()
    sma_slow = series.rolling(window=slow).mean()
    return np.sign(sma_fast - sma_slow).fillna(0)

def compute_price_vs_high(series: pd.Series, window: int = 252) -> pd.Series:
    """Ratio of current price to N-day high."""
    high_n = series.rolling(window=window).max()
    return (series / high_n).fillna(0)

def compute_relative_volume(volume: pd.Series, window: int = 10) -> pd.Series:
    """Volume relative to recent average."""
    vol_sma = volume.rolling(window=window).mean()
    # Add small epsilon to avoid div by zero
    return (volume / (vol_sma + 1e-6)).fillna(0)


def compute_log_return(df: pd.DataFrame) -> pd.Series:
    """Log return: log(Close_t / Close_{t-1})"""
    return np.log(df['Close'] / df['Close'].shift(1))


def compute_volume_change(df: pd.DataFrame) -> pd.Series:
    """Volume change: (Volume_t - Volume_{t-1}) / Volume_{t-1}"""
    return df['Volume'].pct_change()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Args:
        series: Price series
        period: RSI period
    
    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    MACD indicator.
    
    Returns:
        (macd, macd_signal, macd_hist)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    Bollinger Bands.
    
    Returns:
        (bb_mid, bb_std, bb_zscore, bb_width)
    """
    bb_mid = series.rolling(window=period).mean()
    bb_std = series.rolling(window=period).std()
    
    bb_zscore = (series - bb_mid) / bb_std
    bb_width = (num_std * bb_std) / bb_mid
    
    return bb_mid, bb_std, bb_zscore, bb_width


def compute_volatility(log_returns: pd.Series, period: int = 21) -> pd.Series:
    """Rolling volatility from log returns."""
    return log_returns.rolling(window=period).std()


def compute_sma_zscore(series: pd.Series, sma_period: int, zscore_period: int = 20) -> pd.Series:
    """
    Z-score of price relative to SMA.
    
    Args:
        series: Price series
        sma_period: SMA period
        zscore_period: Rolling period for z-score calculation
    
    Returns:
        Z-score
    """
    sma = series.rolling(window=sma_period).mean()
    deviation = series - sma
    
    mean_dev = deviation.rolling(window=zscore_period).mean()
    std_dev = deviation.rolling(window=zscore_period).std()
    
    zscore = (deviation - mean_dev) / std_dev
    return zscore


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smoothed indicators
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx


def compute_trend_persistence(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Counts consecutive days of movement in the same direction.
    Positive for consecutive up days, negative for consecutive down days.
    """
    returns = df['Close'].pct_change()
    
    # +1 for up, -1 for down, 0 for flat
    directions = np.sign(returns).fillna(0)
    
    persistence = pd.Series(index=df.index, dtype=float).fillna(0)
    
    # We can use a rolling apply or a faster iterative approach
    # Since window is small, iterative loop or grouped approach is fine
    # But for vectorization, we can do a trick with groups
    
    # Simple rolling sum of sign doesn't capture "consecutive" strictly
    # Correct approach:
    # If today is UP (+1), persistence = prev_persistence + 1 (if prev > 0) else 1
    # If today is DOWN (-1), persistence = prev_persistence - 1 (if prev < 0) else -1
    
    curr_streak = 0
    streaks = []
    
    for direction in directions:
        if direction > 0:
            curr_streak = curr_streak + 1 if curr_streak > 0 else 1
        elif direction < 0:
            curr_streak = curr_streak - 1 if curr_streak < 0 else -1
        else:
            curr_streak = 0
        streaks.append(curr_streak)
        
    return pd.Series(streaks, index=df.index)



# ============== NEW INDICATORS ==============

def compute_on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).
    Measures buying and selling pressure as cumulative indicator.
    """
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv


def compute_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volume Weighted Average Price.
    Typical price weighted by volume over period.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
    return vwap


def compute_keltner_channels(
    df: pd.DataFrame,
    period: int = 20,
    atr_multiple: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels for volatility-based support/resistance.
    
    Returns:
        (upper, middle, lower) channels
    """
    ema = df['Close'].ewm(span=period).mean()
    atr = compute_atr(df, period)
    
    upper = ema + (atr_multiple * atr)
    lower = ema - (atr_multiple * atr)
    
    return upper, ema, lower


def compute_chaikin_oscillator(df: pd.DataFrame) -> pd.Series:
    """
    Chaikin Oscillator for accumulation/distribution.
    Detects money flow into/out of security.
    """
    # Money Flow Multiplier
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    # Avoid division by zero
    range_hl = high - low
    range_hl = range_hl.replace(0, 1e-10)
    
    mfm = ((close - low) - (high - close)) / range_hl
    mfv = mfm * volume
    
    # Accumulation/Distribution Line
    adl = mfv.cumsum()
    
    # Chaikin Oscillator = EMA(3) - EMA(10) of ADL
    chaikin = adl.ewm(span=3).mean() - adl.ewm(span=10).mean()
    
    return chaikin


def compute_money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI).
    Volume-weighted RSI.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    # Positive and negative money flow
    tp_diff = typical_price.diff()
    positive_mf = money_flow.where(tp_diff > 0, 0)
    negative_mf = money_flow.where(tp_diff < 0, 0)
    
    # Money flow ratio
    positive_sum = positive_mf.rolling(period).sum()
    negative_sum = negative_mf.rolling(period).sum()
    
    mf_ratio = positive_sum / (negative_sum + 1e-10)
    mfi = 100 - (100 / (1 + mf_ratio))
    
    return mfi


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market microstructure features for institutional activity detection.
    
    Features:
    - illiquidity: Amihud illiquidity ratio
    - kyle_lambda: Price impact per unit volume
    - hl_spread: High-Low spread estimator
    - clv: Close location value
    - vpin: Volume-synchronized probability of informed trading (simplified)
    """
    result = pd.DataFrame(index=df.index)
    
    # Amihud illiquidity ratio: |return| / volume
    log_ret = df['Close'].pct_change().abs()
    result['illiquidity'] = log_ret / (df['Volume'] + 1e-10)
    
    # Kyle's lambda proxy: |return| / sqrt(volume)
    result['kyle_lambda'] = log_ret / (np.sqrt(df['Volume']) + 1e-10)
    
    # High-Low spread estimator (Corwin-Schultz simplified)
    hl_ratio = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2 + 1e-10)
    result['hl_spread'] = hl_ratio.rolling(5).mean()
    
    # Close location value (where close is in the day's range)
    range_hl = df['High'] - df['Low']
    range_hl = range_hl.replace(0, 1e-10)
    result['clv'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / range_hl
    
    # Volume z-score (unusual volume detection)
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    result['volume_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-10)
    
    # Trade intensity (volume / volatility)
    volatility = log_ret.rolling(10).std()
    result['trade_intensity'] = df['Volume'] / (volatility + 1e-10)
    
    return result


def compute_stochastic_oscillator(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    
    Returns:
        (k_line, d_line)
    """
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    
    k_line = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    d_line = k_line.rolling(d_period).mean()
    
    return k_line, d_line


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R oscillator.
    """
    high_max = df['High'].rolling(period).max()
    low_min = df['Low'].rolling(period).min()
    
    wr = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)
    
    return wr


def compute_commodity_channel_index(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    return cci


def compute_parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse).
    Simplified implementation.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    sar = close.copy()
    ep = low.iloc[0]
    af = af_start
    trend = 1  # 1 = uptrend, -1 = downtrend
    
    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            
            if low.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            elif high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_increment, af_max)
        else:
            sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
            
            if high.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            elif low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_increment, af_max)
    
    return sar


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a price DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with all indicator columns added
    """
    result = df.copy()
    
    # Basic returns
    result['log_return'] = compute_log_return(df)
    result['volume_change'] = compute_volume_change(df)
    result['quarterly_return'] = df['Close'].pct_change(60).fillna(0)
    
    # ============== INTRADAY FEATURES ==============
    
    # Overnight return: gap between yesterday's close and today's open
    result['overnight_return'] = np.log(df['Open'] / df['Close'].shift(1)).fillna(0)
    
    # Intraday range: (High - Low) / Close, measures daily volatility
    result['intraday_range'] = ((df['High'] - df['Low']) / df['Close']).fillna(0)
    
    # Body ratio: how much of the candle is body vs wick
    # +1 = full bullish candle, -1 = full bearish candle, 0 = doji
    candle_range = (df['High'] - df['Low']).replace(0, 1e-10)
    result['body_ratio'] = ((df['Close'] - df['Open']) / candle_range).fillna(0)
    
    # Upper wick ratio: distance from close/open (whichever is higher) to high
    upper_body = df[['Close', 'Open']].max(axis=1)
    result['upper_wick_ratio'] = ((df['High'] - upper_body) / candle_range).fillna(0)
    
    # Gap direction: 1 if gap up, -1 if gap down, 0 if flat
    result['gap_direction'] = np.sign(result['overnight_return'])
    
    # Consecutive gap direction (streak of gap-ups or gap-downs)
    result['gap_streak'] = result['gap_direction'].groupby(
        (result['gap_direction'] != result['gap_direction'].shift()).cumsum()
    ).cumcount() + 1
    result['gap_streak'] = result['gap_streak'] * result['gap_direction']
    
    # RSI variants
    result['rsi_6'] = compute_rsi(df['Close'], period=6)
    result['rsi_14'] = compute_rsi(df['Close'], period=14)
    result['rsi_delta'] = result['rsi_14'].diff().fillna(0)
    
    # MACD family
    macd, macd_signal, macd_hist = compute_macd(df['Close'])
    result['macd'] = macd
    result['macd_signal'] = macd_signal
    result['macd_hist'] = macd_hist
    result['macd_momentum'] = result['macd_hist'].diff().fillna(0)
    
    # Bollinger Bands
    bb_mid, bb_std, bb_zscore, bb_width = compute_bollinger_bands(df['Close'])
    result['bb_mid'] = bb_mid
    result['bb_std'] = bb_std
    result['bb_zscore'] = bb_zscore.fillna(0)
    result['bb_width'] = bb_width
    result['zscore_velocity'] = result['bb_zscore'].diff().fillna(0)
    
    # Volatility variants
    result['volatility_7'] = compute_volatility(result['log_return'], period=7)
    result['volatility_21'] = compute_volatility(result['log_return'], period=21)
    result['volatility_60'] = compute_volatility(result['log_return'], period=60)
    
    # Volume expansion: ratio of short-term to long-term volatility
    eps = 1e-8
    result['vol_expansion'] = result['volatility_7'] / (result['volatility_21'] + eps)
    result['vol_acceleration'] = result['vol_expansion'].diff().fillna(0)
    
    # SMA-based features
    result['sma_20'] = df['Close'].rolling(window=20).mean()
    result['sma_50'] = df['Close'].rolling(window=50).mean()
    result['sma_100'] = df['Close'].rolling(window=100).mean()
    result['sma_200'] = df['Close'].rolling(window=200).mean()
    
    result['sma_20_z'] = compute_sma_zscore(df['Close'], sma_period=20).fillna(0)
    result['sma_50_z'] = compute_sma_zscore(df['Close'], sma_period=50).fillna(0)
    result['sma_100_z'] = compute_sma_zscore(df['Close'], sma_period=100).fillna(0)
    result['sma_200_z'] = compute_sma_zscore(df['Close'], sma_period=200).fillna(0)
    
    # ADX and derivatives
    result['adx'] = compute_adx(df, period=14)
    result['trend_strength'] = result['adx'] / 100.0
    
    # Regime indicator
    result['regime_sma'] = (result['sma_20'] > result['sma_50']).astype(float)
    
    # Trend Persistence
    result['trend_persistence'] = compute_trend_persistence(df)
    
    # Price-volume divergence
    # Simple momentum divergence: price_mom_5d - vol_mom_5d
    price_mom_5d = df["Close"].pct_change(5)
    vol_mom_5d = df["Volume"].pct_change(5)
    result['pv_divergence'] = (price_mom_5d - vol_mom_5d).fillna(0)
    
    # Fibonacci Retracement
    fib_df = compute_fibonacci_levels(df, lookback=60)
    # Join fib columns
    fib_cols = [c for c in fib_df.columns if c not in result.columns]
    result = pd.concat([result, fib_df[fib_cols]], axis=1)
    
    # NEW: Momentum & Volatility
    result['atr_14'] = compute_atr(df, period=14)
    result['atr_28'] = compute_atr(df, period=28)
    # Normalized ATR (Volatility relative to price)
    result['natr'] = (result['atr_14'] / result['Close']) * 100
    
    # RSI Velocity
    result['rsi_velocity'] = result['rsi_14'].diff(5).fillna(0)
    
    # NEW: Macro & Volatility Regime
    result['macro_proxy'] = compute_macro_proxy(df)
    result['atr_change'] = compute_atr_change(df, period=21)
    
    # NEW: Rolling Sentiment Features
    if 'sentiment_mean' in result.columns and 'sentiment_count' in result.columns:
        # 5-day rolling average sentiment
        result['sentiment_rolling_5d'] = result['sentiment_mean'].rolling(window=5, min_periods=1).mean()
        
        # Sentiment momentum (current - 5d avg)
        result['sentiment_momentum'] = result['sentiment_mean'] - result['sentiment_rolling_5d']
        
        # Sentiment spike: High volume of news (> 2 std dev from 21d mean)
        sent_count_rolling = result['sentiment_count'].rolling(window=21, min_periods=5)
        sent_count_mean = sent_count_rolling.mean()
        sent_count_std = sent_count_rolling.std()
        
        # Default to 0 if std/mean is NaN
        sent_count_mean = sent_count_mean.fillna(0)
        sent_count_std = sent_count_std.fillna(1e-6)
        
        z_score_count = (result['sentiment_count'] - sent_count_mean) / sent_count_std
        result['sentiment_spike'] = (z_score_count > 2.0).astype(int)
        
        # NEW Long-Horizon Sentiment Features
        result['sentiment_rolling_10d'] = result['sentiment_mean'].rolling(window=10, min_periods=1).mean()
        result['sentiment_rolling_21d'] = result['sentiment_mean'].rolling(window=21, min_periods=1).mean()
        
        # Sentiment Trend (Slope of 10d rolling)
        # We can approximate slope by diff(5) of the 10d avg
        result['sentiment_trend'] = result['sentiment_rolling_10d'].diff(5).fillna(0)
    
    # NEW: Cross-Asset Proxies
    result = compute_cross_asset_features(result)
    
    # NEW (Phase 2): Long-Horizon Trend Features
    result['sma_10_20_cross'] = compute_sma_cross(df['Close'], 10, 20)
    result['sma_20_50_cross'] = compute_sma_cross(df['Close'], 20, 50)
    result['sma_50_200_cross'] = compute_sma_cross(df['Close'], 50, 200)
    result['price_vs_52w_high'] = compute_price_vs_high(df['Close'], window=252)
    result['relative_volume_10d'] = compute_relative_volume(df['Volume'], window=10)
    
    # ============== RELATIVE STRENGTH ==============
    # 5-day momentum: stock's own 5-day return
    result['momentum_5d'] = df['Close'].pct_change(5).fillna(0)
    result['momentum_10d'] = df['Close'].pct_change(10).fillna(0)
    
    # Relative strength vs own average (self-relative)
    # How today's momentum compares to rolling 21d average momentum
    avg_mom = result['momentum_5d'].rolling(21).mean()
    result['relative_momentum'] = (result['momentum_5d'] - avg_mom).fillna(0)
    
    # ============== NEW INDICATORS (V2) ==============
    
    # On-Balance Volume
    result['obv'] = compute_on_balance_volume(df)
    result['obv_sma'] = result['obv'].rolling(20).mean()
    result['obv_divergence'] = (result['obv'] - result['obv_sma']) / (result['obv_sma'].abs() + 1e-10)
    
    # VWAP
    result['vwap_20'] = compute_vwap(df, period=20)
    result['vwap_10'] = compute_vwap(df, period=10)
    result['price_vs_vwap'] = (df['Close'] - result['vwap_20']) / result['vwap_20']
    
    # Keltner Channels
    kc_upper, kc_middle, kc_lower = compute_keltner_channels(df, period=20, atr_multiple=2.0)
    result['kc_upper'] = kc_upper
    result['kc_lower'] = kc_lower
    result['kc_width'] = (kc_upper - kc_lower) / kc_middle
    result['price_vs_kc'] = (df['Close'] - kc_middle) / ((kc_upper - kc_lower) / 2 + 1e-10)
    
    # Chaikin Oscillator
    result['chaikin_osc'] = compute_chaikin_oscillator(df)
    result['chaikin_signal'] = (result['chaikin_osc'] > 0).astype(float)
    
    # Money Flow Index
    result['mfi_14'] = compute_money_flow_index(df, period=14)
    result['mfi_oversold'] = (result['mfi_14'] < 20).astype(float)
    result['mfi_overbought'] = (result['mfi_14'] > 80).astype(float)
    
    # Stochastic Oscillator
    stoch_k, stoch_d = compute_stochastic_oscillator(df, k_period=14, d_period=3)
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    result['stoch_cross'] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(float)
    
    # Williams %R
    result['williams_r'] = compute_williams_r(df, period=14)
    
    # Commodity Channel Index
    result['cci'] = compute_commodity_channel_index(df, period=20)
    
    # Microstructure Features
    micro_features = compute_microstructure_features(df)
    for col in micro_features.columns:
        result[col] = micro_features[col]
    
    # Fill NaN values for new features
    new_cols = ['obv', 'obv_sma', 'obv_divergence', 'vwap_20', 'vwap_10', 'price_vs_vwap',
                'kc_upper', 'kc_lower', 'kc_width', 'price_vs_kc', 'chaikin_osc', 'chaikin_signal',
                'mfi_14', 'mfi_oversold', 'mfi_overbought', 'stoch_k', 'stoch_d', 'stoch_cross',
                'williams_r', 'cci', 'illiquidity', 'kyle_lambda', 'hl_spread', 'clv', 
                'volume_zscore', 'trade_intensity']
    for col in new_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


def compute_cross_sectional_features(
    df: pd.DataFrame,
    group_col: str = 'symbol',
    date_col: str = 'date',
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    return_col: str = 'log_return',
    lookback_periods: List[int] = [5, 10, 21]
) -> pd.DataFrame:
    """
    Compute cross-sectional features that compare a stock to its peers.
    
    These features capture relative strength, momentum, and valuation
    compared to other stocks in the universe on the same date.
    
    Args:
        df: DataFrame with multiple stocks (must have 'symbol', 'date' columns)
        group_col: Column name for stock identifier
        date_col: Column name for date
        price_col: Column name for price
        volume_col: Column name for volume
        return_col: Column name for returns
        lookback_periods: Periods for computing relative metrics
    
    Returns:
        DataFrame with cross-sectional features added
    """
    result = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
        result[date_col] = pd.to_datetime(result[date_col])
    
    # Sort by date
    result = result.sort_values([date_col, group_col])
    
    # ============== RETURN RANKING ==============
    # For each date, rank stocks by their returns
    for period in lookback_periods:
        # Compute cumulative return over period
        cum_return = result.groupby(group_col)[price_col].pct_change(period)
        result[f'cum_return_{period}d'] = cum_return
        
        # Cross-sectional rank (0-1, where 1 is highest return)
        result[f'return_rank_{period}d'] = result.groupby(date_col)[f'cum_return_{period}d'].rank(
            method='average', pct=True
        )
        
        # Z-score vs peers
        result[f'return_zscore_{period}d'] = result.groupby(date_col)[f'cum_return_{period}d'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        
        # Percentile bins (quintiles)
        result[f'return_quintile_{period}d'] = result.groupby(date_col)[f'cum_return_{period}d'].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        )
    
    # ============== VOLUME RANKING ==============
    # Relative volume vs peers
    result['volume_rank'] = result.groupby(date_col)[volume_col].rank(
        method='average', pct=True
    )
    result['volume_zscore'] = result.groupby(date_col)[volume_col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    
    # ============== VOLATILITY RANKING ==============
    # Compute rolling volatility first
    for period in [7, 21]:
        vol = result.groupby(group_col)[return_col].transform(
            lambda x: x.rolling(period, min_periods=1).std()
        )
        result[f'volatility_{period}d'] = vol
        
        # Volatility rank (high rank = high volatility)
        result[f'volatility_rank_{period}d'] = result.groupby(date_col)[f'volatility_{period}d'].rank(
            method='average', pct=True
        )
    
    # ============== RELATIVE STRENGTH ==============
    # Price vs market average (equal-weighted)
    market_avg = result.groupby(date_col)[price_col].transform('mean')
    result['price_vs_market'] = result[price_col] / market_avg - 1
    
    # Relative strength vs market (rolling)
    for period in [10, 21]:
        # Stock's return
        stock_return = result.groupby(group_col)[price_col].pct_change(period)
        # Market's return (equal-weighted average of all stocks)
        market_return = result.groupby(date_col)[price_col].transform(
            lambda x: x.pct_change(period) if len(x) > 1 else 0
        )
        # Actually need to compute this differently
        # Simplified: relative strength = stock price / market avg price, then pct change
        rs = result['price_vs_market']
        result[f'relative_strength_{period}d'] = rs.groupby(result[group_col]).transform(
            lambda x: x.pct_change(period)
        )
    
    # ============== MOMENTUM DECILE ==============
    # Classic momentum ranking (12-1 month momentum)
    # Simplified to available periods
    result['momentum_score'] = (
        result['return_rank_21d'] * 0.5 +
        result['return_rank_10d'] * 0.3 +
        result['return_rank_5d'] * 0.2
    )
    result['momentum_decile'] = result.groupby(date_col)['momentum_score'].transform(
        lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
    )
    
    # ============== MEAN REVERSION SIGNAL ==============
    # Stocks in bottom decile of returns may be due for reversal
    result['mean_reversion_signal'] = (
        (result['return_quintile_21d'] == 0) & 
        (result['return_quintile_5d'] == 0)
    ).astype(float)
    
    # ============== LIQUIDITY RANKING ==============
    # Volume * Price as proxy for dollar volume
    dollar_volume = result[volume_col] * result[price_col]
    result['dollar_volume_rank'] = result.groupby(date_col)[dollar_volume.name if dollar_volume.name else 'dollar_volume'].rank(
        method='average', pct=True
    )
    
    # ============== CORRELATION WITH MARKET ==============
    # Rolling correlation with market return
    market_return = result.groupby(date_col)[return_col].transform('mean')
    for period in [21]:
        result[f'market_corr_{period}d'] = result.groupby(group_col).apply(
            lambda g: g[return_col].rolling(period, min_periods=10).corr(market_return.loc[g.index])
        ).reset_index(level=0, drop=True)
    
    # ============== BETA ESTIMATION ==============
    # Rolling beta = Cov(r_i, r_m) / Var(r_m)
    for period in [21]:
        market_var = market_return.groupby(result[group_col]).transform(
            lambda x: x.rolling(period, min_periods=10).var()
        )
        cov = result.groupby(group_col).apply(
            lambda g: g[return_col].rolling(period, min_periods=10).apply(
                lambda x: np.cov(x, market_return.loc[x.index])[0, 1] if len(x) > 1 else 0
            )
        ).reset_index(level=0, drop=True)
        result[f'beta_{period}d'] = cov / (market_var + 1e-10)
    
    # Fill NaN values
    cs_cols = [c for c in result.columns if any(x in c for x in ['rank', 'zscore', 'quintile', 'decile', 'signal', 'vs_market', 'strength', 'corr', 'beta'])]
    for col in cs_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    
    return result


def add_cross_sectional_to_panel(
    panel_df: pd.DataFrame,
    feature_cols: List[str] = ['log_return', 'Volume', 'rsi_14', 'bb_zscore', 'volatility_21']
) -> pd.DataFrame:
    """
    Add cross-sectional statistics for a set of features.
    
    For each feature, adds:
    - {feature}_cs_mean: Cross-sectional mean
    - {feature}_cs_std: Cross-sectional std
    - {feature}_cs_zscore: Z-score vs peers
    
    Args:
        panel_df: DataFrame with columns ['symbol', 'date', ...features]
        feature_cols: List of feature columns to compute cross-sectional stats for
    
    Returns:
        DataFrame with cross-sectional stats added
    """
    result = panel_df.copy()
    
    if 'symbol' not in result.columns or 'date' not in result.columns:
        return result
    
    for col in feature_cols:
        if col not in result.columns:
            continue
        
        # Cross-sectional mean and std
        cs_mean = result.groupby('date')[col].transform('mean')
        cs_std = result.groupby('date')[col].transform('std')
        
        result[f'{col}_cs_mean'] = cs_mean
        result[f'{col}_cs_std'] = cs_std.fillna(0)
        result[f'{col}_cs_zscore'] = (result[col] - cs_mean) / (cs_std + 1e-10)
        result[f'{col}_cs_zscore'] = result[f'{col}_cs_zscore'].fillna(0)
    
    return result
