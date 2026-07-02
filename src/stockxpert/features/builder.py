"""
Feature builder that computes indicators and targets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from .indicators import compute_all_indicators
from .event_calendar import add_event_flags

logger = logging.getLogger("stockxpert.features.builder")


class FeatureBuilder:
    """
    Builds complete feature set including indicators and targets.
    """
    
    def __init__(self, horizons: List[int]):
        """
        Args:
            horizons: List of prediction horizons (e.g., [1, 3, 5, 7, 10])
        """
        self.horizons = horizons
    
    def build_features(
        self, 
        price_data: Dict[str, pd.DataFrame],
        fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
        is_inference: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all indicators and targets for all symbols.
        
        Args:
            price_data: Dict of symbol -> DataFrame with OHLCV + sentiment
            fundamental_data: Optional dict of symbol -> DataFrame with fundamentals
            is_inference: If True, skip target computation and don't drop rows based on targets
        
        Returns:
            Dict of symbol -> DataFrame with indicators + targets
        """
        results = {}
        
        for symbol, df in price_data.items():
            logger.info(f"Building features for {symbol}")
            
            # Compute indicators
            df_features = compute_all_indicators(df)
            
            # Merge fundamentals if provided
            if fundamental_data and symbol in fundamental_data:
                fund_df = fundamental_data[symbol]
                if not fund_df.empty:
                    # Ensure both indices are tz-naive for join
                    if df_features.index.tz is not None:
                        df_features.index = df_features.index.tz_localize(None)
                    if hasattr(fund_df.index, 'tz') and fund_df.index.tz is not None:
                        fund_df.index = fund_df.index.tz_localize(None)
                    
                    # Drop any overlapping columns from fund_df to avoid conflict
                    overlap_cols = fund_df.columns.intersection(df_features.columns)
                    if len(overlap_cols) > 0:
                        logger.debug(f"Dropping overlapping columns from fundamentals: {list(overlap_cols)}")
                        fund_df = fund_df.drop(columns=overlap_cols)
                    
                    # Merge on index (Date)
                    # Use join with how='left' to preserve price dates
                    if not fund_df.empty:
                        df_features = df_features.join(fund_df, how='left')
                        # Forward fill to handle sparse fundamental dates
                        fund_cols = fund_df.columns
                        df_features[fund_cols] = df_features[fund_cols].ffill().bfill()
            
            # Ensure fundamental columns exist (for robustness against missing data)
            if 'pe_ratio' not in df_features.columns:
                df_features['pe_ratio'] = np.nan
            if 'eps_trailing' not in df_features.columns:
                df_features['eps_trailing'] = np.nan
            
            # Final fill for any remaining NaNs
            df_features = df_features.fillna(0.0)
            
            # Add event flags
            df_features = add_event_flags(df_features)
            
            # Always compute vol_ref (uses only past data, needed for Z->return conversion)
            eps = 1e-8
            if 'log_return' not in df_features.columns:
                df_features['log_return'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
            df_features['vol_ref'] = df_features['log_return'].rolling(10).std() + eps
            
            # Compute targets - ONLY if not inference
            if not is_inference:
                df_features = self._compute_targets(df_features)
            
            # Drop warm-up rows
            df_features = self._drop_warmup(df_features, is_inference=is_inference)
            
            results[symbol] = df_features
            if not is_inference:
                 logger.info(f"  {symbol}: {len(df_features)} valid rows after warm-up")
        
        return results
    
    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multi-horizon targets as Z-score-normalized returns.

        The target contract used across the repo applies a short centered
        smoothing window to future log returns before volatility
        normalization. Keeping this logic stable matters because training
        runs, analysis scripts, and tests all assume the same target
        definition.
        
        Args:
            df: DataFrame with Close column
        
        Returns:
            DataFrame with added target columns [target_h1, target_h3, ...]
            and 'vol_ref' column.
        """
        result = df.copy()
        eps = 1e-8
        
        # 0. Rolling volatility reference for normalization
        # Use log_return computed in indicators
        if 'log_return' not in result.columns:
            result['log_return'] = np.log(result['Close'] / result['Close'].shift(1))
            
        rolling_std = result['log_return'].rolling(10).std() + eps
        result['vol_ref'] = rolling_std
        
        log_close = np.log(result['Close'])
        
        for h in self.horizons:
            # 1. Raw Δlog = log(Close_{t+h}) - log(Close_t)
            log_close_future = log_close.shift(-h)
            raw_dlog = log_close_future - log_close
            
            # 2. Smooth future returns with the established centered window.
            # The 3-step centered mean is the repo's current target contract
            # and is validated by tests.
            smooth_window = max(3, min(h, 7))
            smoothed_dlog = raw_dlog.rolling(smooth_window, center=True).mean().fillna(raw_dlog)
            
            # 3. Z-score Normalization
            # z = smoothed_return / current_volatility
            # This makes targets stationary across high/low vol regimes
            z_dlog = smoothed_dlog / rolling_std
            
            # Clip extreme values (e.g. > 5 sigma events)
            z_dlog = z_dlog.clip(-5, 5)
            
            result[f'target_h{h}'] = z_dlog

            # --- NEW: Level Targets (Support/Resistance/Target) ---
            # Future High (Resistance) and Future Low (Support) within the horizon
            future_high = result['High'].shift(-h).rolling(window=h, min_periods=1).max().shift(h-1)
            future_low = result['Low'].shift(-h).rolling(window=h, min_periods=1).min().shift(h-1)
            # Wait, easier way for lookahead rolling:
            # Shift(-h) followed by reverse rolling max.
            # actually rolling works forward.
            # Let's use simpler shift for the value at exactly h days, 
            # and a rolling min/max of the window [t+1, t+h].
            
            f_h = result['High'].iloc[::-1].rolling(h).max().iloc[::-1].shift(-h)
            f_l = result['Low'].iloc[::-1].rolling(h).min().iloc[::-1].shift(-h)
            f_c = result['Close'].shift(-h)
            
            # Normalize levels: (Price_future - Price_now) / (Price_now * Volatility)
            # This follows the same Z-score logic as magnitude
            result[f'level_target_h{h}_res'] = np.log(f_h / result['Close']) / rolling_std
            result[f'level_target_h{h}_sup'] = np.log(f_l / result['Close']) / rolling_std
            result[f'level_target_h{h}_tgt'] = np.log(f_c / result['Close']) / rolling_std
            
            # Clip levels
            for level in ['res', 'sup', 'tgt']:
                result[f'level_target_h{h}_{level}'] = result[f'level_target_h{h}_{level}'].clip(-10, 10)
        
        return result
    
    def _drop_warmup(self, df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
        """
        Drop initial rows where indicators or targets are NaN or Inf.
        
        Ensures all features and targets are valid.
        """
        # Find first row where all selected features + targets are non-NaN
        # We'll be conservative and drop NaN in any column (can be optimized)
        
        # Critical columns to check
        critical_cols = [
            'log_return', 'volume_change', 'rsi_6', 'rsi_14',
            'macd', 'macd_signal', 'macd_hist', 'macd_momentum',
            'bb_zscore', 'bb_width', 'volatility_7', 'volatility_21', 'volatility_60',
            'sma_20_z', 'sma_50_z', 'adx', 'trend_strength', 'regime_sma',
            'vol_expansion', 'vol_acceleration', 'pv_divergence', 'rsi_delta',
            'zscore_velocity',
            'fib_distance_to_nearest', 'fib_relative_position', 'fib_zone',
            'atr_14', 'natr', 'rsi_velocity'
        ]
        
        # Add targets ONLY if not inference
        if not is_inference:
            target_cols = [f'target_h{h}' for h in self.horizons]
            critical_cols.extend(target_cols)
        
        # Filter to existing columns (some might not exist if error)
        critical_cols = [c for c in critical_cols if c in df.columns]
        
        before_len = len(df)
        
        # Replace inf with nan
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with any NaN in critical columns
        df_clean = df_clean.dropna(subset=critical_cols)
        after_len = len(df_clean)
        
        # if before_len > after_len:
            # logger.debug(f"Dropped {before_len - after_len} warm-up/invalid rows")
        
        return df_clean
