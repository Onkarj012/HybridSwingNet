"""
Sample construction for multi-scale tensors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("stockxpert.dataset.make_samples")


@dataclass
class Sample:
    """
    A single training sample with multi-scale inputs.
    
    Shapes:
        X_short: (win_short, n_short_features)
        X_mid: (win_mid, n_mid_features)
        X_long: (win_long, n_long_features)
        X_context: (n_context_features,)
        X_sentiment: (n_sentiment_features,)
        y: (n_horizons,) - Magnitude targets (normalized log return)
        y_zone: (n_horizons,) - Target zone indices (0-6)
    
    Metadata:
        date: Date of sample (t)
        symbol: Stock symbol
        stock_idx: Integer index for embedding
        close_t: Close price at time t
    """
    X_short: np.ndarray
    X_mid: np.ndarray
    X_long: np.ndarray
    X_context: np.ndarray
    X_sentiment: np.ndarray
    y: np.ndarray
    y_zone: np.ndarray
    y_levels: np.ndarray
    
    y_trend: float
    
    date: pd.Timestamp
    symbol: str
    stock_idx: int
    close_t: float
    vol_ref: float


class SampleBuilder:
    """
    Builds samples from feature DataFrames.
    """
    
    def __init__(
        self,
        window_short: int,
        window_mid: int,
        window_long: int,
        short_features: List[str],
        mid_features: List[str],
        long_features: List[str],
        context_features: List[str],
        sentiment_features: List[str],
        horizons: List[int]
    ):
        """
        Args:
            window_short: Short window length (e.g., 7)
            window_mid: Mid window length (e.g., 21)
            window_long: Long window length (e.g., 60)
            short_features: List of feature names for short window
            mid_features: List of feature names for mid window
            long_features: List of feature names for long window
            context_features: List of feature names for context
            sentiment_features: List of feature names for sentiment
            horizons: List of target horizons
        """
        self.window_short = window_short
        self.window_mid = window_mid
        self.window_long = window_long
        self.short_features = short_features
        self.mid_features = mid_features
        self.long_features = long_features
        self.context_features = context_features
        self.sentiment_features = sentiment_features
        self.horizons = horizons
        
        # Max window determines minimum required history
        self.max_window = max(window_short, window_mid, window_long)
        
        logger.info(f"SampleBuilder initialized:")
        logger.info(f"  Windows: short={window_short}, mid={window_mid}, long={window_long}")
        logger.info(f"  Features: short={len(short_features)}, mid={len(mid_features)}, "
                   f"long={len(long_features)}, context={len(context_features)}, "
                   f"sentiment={len(sentiment_features)}")
    
    def build_samples(
        self,
        feature_data: Dict[str, pd.DataFrame],
        symbol_map: Dict[str, int]
    ) -> List[Sample]:
        """
        Build samples from feature DataFrames.
        
        Args:
            feature_data: Dict of symbol -> DataFrame with features + targets
            symbol_map: Dict of symbol -> integer index for embedding
        
        Returns:
            List of Sample objects
        """
        all_samples = []
        
        for symbol, df in feature_data.items():
            stock_idx = symbol_map[symbol]
            
            # Build samples for this symbol
            symbol_samples = self._build_samples_for_symbol(df, symbol, stock_idx)
            all_samples.extend(symbol_samples)
            
            logger.info(f"Built {len(symbol_samples)} samples for {symbol}")
        
        logger.info(f"Total samples: {len(all_samples)}")
        return all_samples
    
    def _build_samples_for_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        stock_idx: int
    ) -> List[Sample]:
        """Build samples for a single symbol."""
        samples = []
        
        # Start from max_window to ensure we have enough history
        # And ensure we have future data for targets (drop last max(horizons) rows)
        max_horizon = max(self.horizons)
        end_idx = len(df) - max_horizon
        
        for t in range(self.max_window, end_idx):
            try:
                sample = self._build_single_sample(df, t, symbol, stock_idx)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to build sample for {symbol} at index {t}: {e}")
                continue
        
        return samples
    
    def _build_single_sample(
        self,
        df: pd.DataFrame,
        t: int,
        symbol: str,
        stock_idx: int
    ) -> Sample:
        """
        Build a single sample at time index t.
        """
        # Extract windows
        short_start = t - self.window_short + 1
        X_short = df.iloc[short_start:t+1][self.short_features].values
        
        mid_start = t - self.window_mid + 1
        X_mid = df.iloc[mid_start:t+1][self.mid_features].values
        
        long_start = t - self.window_long + 1
        X_long = df.iloc[long_start:t+1][self.long_features].values
        
        # Context & Sentiment
        X_context = df.iloc[t][self.context_features].values
        
        # Handle case where sentiment features might strictly not exist (nan padding?)
        # But we assume they exist if passed in config
        if self.sentiment_features:
            # Check if columns exist, else zero
            valid_cols = [c for c in self.sentiment_features if c in df.columns]
            if len(valid_cols) == len(self.sentiment_features):
                X_sentiment = df.iloc[t][self.sentiment_features].values
            else:
                 X_sentiment = np.zeros(len(self.sentiment_features))
        else:
            X_sentiment = np.array([])
        
        # Targets: multi-horizon magnitude
        target_cols = [f'target_h{h}' for h in self.horizons]
        y = df.iloc[t][target_cols].values
        
        # Targets: Target Zones (Fib Levels)
        # We need future prices
        y_zone = []
        fib_cols = ['fib_0.0', 'fib_23.6', 'fib_38.2', 'fib_50.0', 'fib_61.8', 'fib_78.6', 'fib_100.0']
        
        # Current Fib levels at time t
        current_fibs = df.iloc[t][fib_cols].values # (7,)
        
        for h in self.horizons:
            # Future price
            future_price = df.iloc[t+h]['Close']
            
            # Find closest level index
            # abs diff
            diffs = np.abs(current_fibs - future_price)
            closest_idx = np.argmin(diffs)
            y_zone.append(closest_idx)
            
        y_zone = np.array(y_zone)
        
        # Targets: Price Levels [Support, Resistance, Target]
        y_levels = []
        for h in self.horizons:
            lvl_res = df.iloc[t][f'level_target_h{h}_res']
            lvl_sup = df.iloc[t][f'level_target_h{h}_sup']
            lvl_tgt = df.iloc[t][f'level_target_h{h}_tgt']
            y_levels.append([lvl_res, lvl_sup, lvl_tgt])
        y_levels = np.array(y_levels)

        # Auxiliary Trend Target (10-day future SMA vs current price)
        # Calculate future 10-day mean: mean(Close[t+1 : t+11])
        # If t+11 exceeds bounds, we are already protected by end_idx in _build_samples_for_symbol
        future_prices = df.iloc[t+1:t+11]['Close'].values
        if len(future_prices) > 0:
            future_sma = np.mean(future_prices)
            current_close = df.iloc[t]['Close']
            y_trend = 1.0 if future_sma > current_close else 0.0
        else:
            y_trend = 0.5 # Default neutral

        # Metadata
        date = df.index[t]
        close_t = df.iloc[t]['Close']
        vol_ref = df.iloc[t]['vol_ref'] if 'vol_ref' in df.columns else 1.0
        
        return Sample(
            X_short=X_short.astype(np.float32),
            X_mid=X_mid.astype(np.float32),
            X_long=X_long.astype(np.float32),
            X_context=X_context.astype(np.float32),
            X_sentiment=X_sentiment.astype(np.float32),
            y=y.astype(np.float32),
            y_zone=y_zone.astype(np.int64),
            y_levels=y_levels.astype(np.float32),
            y_trend=float(y_trend),
            date=date,
            symbol=symbol,
            stock_idx=stock_idx,
            close_t=float(close_t),
            vol_ref=float(vol_ref)
        )
