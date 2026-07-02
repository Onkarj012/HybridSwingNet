"""
Test that targets are correctly aligned with future dates.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stockxpert.features.builder import FeatureBuilder


def test_target_calculation():
    """
    Test that targets correctly compute delta log returns for future horizons.
    """
    # Create mock price DataFrame
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    close_prices = np.exp(np.random.randn(100).cumsum() * 0.01 + 4.0)  # Simulated log-returns
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices,
        'High': close_prices * 1.01,
        'Low': close_prices * 0.99,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    df.set_index('Date', inplace=True)
    
    # Add sentiment (neutral)
    df['sentiment_mean'] = 0.0
    df['sentiment_count'] = 0
    df['sentiment_std'] = 0.0
    
    # Build features
    # NOTE: _compute_targets now requires log_return column for vol reference
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
    fb = FeatureBuilder(horizons=[1, 3, 5])
    df_features = fb._compute_targets(df)
    
    # Calculate expected values manually using the same logic
    rolling_std = df['log_return'].rolling(10).std() + 1e-8
    log_close = np.log(df['Close'])
    
    # Check target_h1
    h = 1
    raw_dlog = log_close.shift(-h) - log_close
    smoothed_dlog = raw_dlog.rolling(3, center=True).mean().fillna(raw_dlog)
    z_dlog = (smoothed_dlog / rolling_std).clip(-5, 5)
    
    for i in range(len(df) - h):
        expected = z_dlog.iloc[i]
        actual = df_features.iloc[i][f'target_h{h}']
        
        if not pd.isna(actual) and not pd.isna(expected):
            np.testing.assert_almost_equal(actual, expected, decimal=6,
                err_msg=f"Target h{h} mismatch at index {i}")
    
    # Check target_h3
    h = 3
    raw_dlog = log_close.shift(-h) - log_close
    smoothed_dlog = raw_dlog.rolling(3, center=True).mean().fillna(raw_dlog)
    z_dlog = (smoothed_dlog / rolling_std).clip(-5, 5)
    
    for i in range(len(df) - h):
        expected = z_dlog.iloc[i]
        actual = df_features.iloc[i][f'target_h{h}']
        
        if not pd.isna(actual) and not pd.isna(expected):
            np.testing.assert_almost_equal(actual, expected, decimal=6,
                err_msg=f"Target h{h} mismatch at index {i}")
    
    print("✓ Target calculations correct")


if __name__ == '__main__':
    test_target_calculation()
    print("\nAll target alignment tests passed!")
