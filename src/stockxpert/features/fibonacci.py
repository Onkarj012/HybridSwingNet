"""
Fibonacci Retracement Calculator.
Auto-detects swing highs and lows to compute key levels.
"""

import pandas as pd
import numpy as np

def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute Fibonacci retracement levels based on recent swing high/low.
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        lookback: Window size to find swing high/low points
        
    Returns:
        DataFrame with added columns:
        - fib_swing_high, fib_swing_low
        - fib_23.6, fib_38.2, fib_50.0, fib_61.8, fib_78.6
        - fib_distance_to_nearest (normalized)
        - fib_zone (categorical 0-4)
    """
    result = df.copy()
    
    # 1. Rolling Max/Min for Swing Points
    # We use a rolling window to find the highest high and lowest low
    result['fib_swing_high'] = df['High'].rolling(window=lookback).max()
    result['fib_swing_low'] = df['Low'].rolling(window=lookback).min()
    
    # Range
    fib_range = result['fib_swing_high'] - result['fib_swing_low']
    
    # 2. Compute Levels
    # Typical Retracement Levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    # Level = Low + (Range * Level_Ratio) if Trend is UP?
    # Or purely range-based division regardless of trend direction for Support/Resistance
    # We'll compute standard levels relative to the range
    
    # Note: Traditionally, retracement direction depends on the trend.
    # Here, we treat them as potential support/resistance zones within the current range.
    
    result['fib_0.0'] = result['fib_swing_low']               # 0% (Low)
    result['fib_23.6'] = result['fib_swing_low'] + fib_range * 0.236
    result['fib_38.2'] = result['fib_swing_low'] + fib_range * 0.382
    result['fib_50.0'] = result['fib_swing_low'] + fib_range * 0.500
    result['fib_61.8'] = result['fib_swing_low'] + fib_range * 0.618
    result['fib_78.6'] = result['fib_swing_low'] + fib_range * 0.786
    result['fib_100.0'] = result['fib_swing_high']            # 100% (High)
    
    # 3. Distance to Nearest Level
    # Useful feature for NN: "Are we close to a key level?"
    close = df['Close']
    
    levels = ['fib_0.0', 'fib_23.6', 'fib_38.2', 'fib_50.0', 'fib_61.8', 'fib_78.6', 'fib_100.0']
    
    # Compute absolute distance to each level
    distances = []
    for level_col in levels:
        distances.append(abs(close - result[level_col]))
    
    # Find min distance
    min_dist = pd.concat(distances, axis=1).min(axis=1)
    
    # Normalize distance by price (percentage distance)
    # e.g. 0.001 means we are 0.1% away from a level
    result['fib_distance_to_nearest'] = min_dist / (close + 1e-8)
    
    # 4. Fibonacci Zone
    # Categorical feature: Which "zone" are we in?
    # 0: < 23.6%
    # 1: 23.6% - 38.2%
    # 2: 38.2% - 50.0%
    # 3: 50.0% - 61.8%
    # 4: 61.8% - 78.6%
    # 5: > 78.6%
    
    # Using np.select or simpler logic
    # We'll use a continuous proxy for zone to be NN-friendly: "Relative Position"
    # 0.0 = Low, 1.0 = High
    result['fib_relative_position'] = (close - result['fib_swing_low']) / (fib_range + 1e-8)
    
    # Discrete zone (optional, maybe better as embedding but float is fine for now)
    conditions = [
        (result['fib_relative_position'] < 0.236),
        (result['fib_relative_position'] < 0.382),
        (result['fib_relative_position'] < 0.500),
        (result['fib_relative_position'] < 0.618),
        (result['fib_relative_position'] < 0.786)
    ]
    choices = [0.0, 1.0, 2.0, 3.0, 4.0]
    # Default is 5.0 (>= 78.6%)
    
    result['fib_zone'] = np.select(conditions, choices, default=5.0)
    
    # Clean up auxiliary columns if needed, but we keep them for analysis
    
    return result
