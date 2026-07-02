"""
Scaling/normalization for multi-scale inputs.

Includes:
- ScalerGroup: Standard scaling for all input types
- VolatilityAdjustedScaler: Scale targets by rolling volatility
- TargetScaler: Scale magnitude targets with optional volatility adjustment
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path
import logging
from .make_samples import Sample

logger = logging.getLogger("stockxpert.dataset.scaling")


class VolatilityAdjustedScaler:
    """
    Scale targets by rolling volatility for better generalization.
    
    Instead of standard scaling, this scales log returns by rolling volatility,
    resulting in targets that represent "standard deviations" rather than raw returns.
    This helps the model generalize across different volatility regimes.
    """
    
    def __init__(self, lookback: int = 21, min_vol: float = 1e-6):
        """
        Args:
            lookback: Rolling window for volatility calculation
            min_vol: Minimum volatility to avoid division by zero
        """
        self.lookback = lookback
        self.min_vol = min_vol
        self.fitted = False
        self.global_std = 1.0  # Fallback global std
        
    def fit(self, targets: np.ndarray, volatilities: Optional[np.ndarray] = None) -> 'VolatilityAdjustedScaler':
        """
        Fit the scaler. Computes global statistics for fallback.
        
        Args:
            targets: (N, H) array of target values
            volatilities: (N,) array of rolling volatilities (optional)
        """
        # Compute global std for normalization
        self.global_std = np.std(targets) + self.min_vol
        self.fitted = True
        return self
    
    def transform(
        self,
        targets: np.ndarray,
        volatilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Scale targets by volatility.
        
        Args:
            targets: (N, H) array of target values
            volatilities: (N,) array of rolling volatilities
        
        Returns:
            Scaled targets in units of standard deviations
        """
        if volatilities is not None:
            # Scale by rolling volatility
            vol_expanded = volatilities[:, np.newaxis]  # (N, 1)
            scaled = targets / (vol_expanded + self.min_vol)
        else:
            # Fallback to global std
            scaled = targets / self.global_std
        
        return scaled.astype(np.float32)
    
    def inverse_transform(
        self,
        scaled_targets: np.ndarray,
        volatilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Inverse transform scaled targets back to original scale.
        
        Args:
            scaled_targets: (N, H) array of scaled targets
            volatilities: (N,) array of rolling volatilities
        
        Returns:
            Targets in original scale
        """
        if volatilities is not None:
            vol_expanded = volatilities[:, np.newaxis]
            return scaled_targets * (vol_expanded + self.min_vol)
        else:
            return scaled_targets * self.global_std
    
    def save(self, path: Path) -> None:
        """Save scaler to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'lookback': self.lookback,
                'min_vol': self.min_vol,
                'global_std': self.global_std,
                'fitted': self.fitted
            }, f)
    
    @staticmethod
    def load(path: Path) -> 'VolatilityAdjustedScaler':
        """Load scaler from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        scaler = VolatilityAdjustedScaler(
            lookback=data['lookback'],
            min_vol=data['min_vol']
        )
        scaler.global_std = data['global_std']
        scaler.fitted = data['fitted']
        return scaler


class TargetScaler:
    """
    Combined scaler for magnitude targets with optional volatility adjustment.
    
    Supports two modes:
    1. Standard scaling (mean=0, std=1)
    2. Volatility-adjusted scaling (targets in std dev units)
    """
    
    def __init__(
        self,
        use_volatility_adjustment: bool = True,
        lookback: int = 21
    ):
        self.use_volatility_adjustment = use_volatility_adjustment
        self.lookback = lookback
        
        self.standard_scaler = StandardScaler()
        self.vol_scaler = VolatilityAdjustedScaler(lookback=lookback) if use_volatility_adjustment else None
        
        self.fitted = False
        
    def fit(
        self,
        targets: np.ndarray,
        volatilities: Optional[np.ndarray] = None
    ) -> 'TargetScaler':
        """
        Fit the target scaler.
        
        Args:
            targets: (N, H) array of target values
            volatilities: (N,) array of rolling volatilities (required if use_volatility_adjustment)
        """
        # Always fit standard scaler as fallback
        self.standard_scaler.fit(targets)
        
        # Fit volatility scaler if enabled
        if self.use_volatility_adjustment and volatilities is not None:
            self.vol_scaler.fit(targets, volatilities)
        
        self.fitted = True
        return self
    
    def transform(
        self,
        targets: np.ndarray,
        volatilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transform targets.
        """
        if self.use_volatility_adjustment and volatilities is not None and self.vol_scaler is not None:
            return self.vol_scaler.transform(targets, volatilities)
        else:
            return self.standard_scaler.transform(targets).astype(np.float32)
    
    def inverse_transform(
        self,
        scaled_targets: np.ndarray,
        volatilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Inverse transform targets.
        """
        if self.use_volatility_adjustment and volatilities is not None and self.vol_scaler is not None:
            return self.vol_scaler.inverse_transform(scaled_targets, volatilities)
        else:
            return self.standard_scaler.inverse_transform(scaled_targets)
    
    def save(self, path: Path) -> None:
        """Save scaler to disk."""
        data = {
            'use_volatility_adjustment': self.use_volatility_adjustment,
            'lookback': self.lookback,
            'standard_scaler': self.standard_scaler,
            'fitted': self.fitted
        }
        
        if self.vol_scaler is not None:
            data['vol_scaler'] = {
                'lookback': self.vol_scaler.lookback,
                'min_vol': self.vol_scaler.min_vol,
                'global_std': self.vol_scaler.global_std
            }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(path: Path) -> 'TargetScaler':
        """Load scaler from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        scaler = TargetScaler(
            use_volatility_adjustment=data['use_volatility_adjustment'],
            lookback=data['lookback']
        )
        scaler.standard_scaler = data['standard_scaler']
        scaler.fitted = data['fitted']
        
        if 'vol_scaler' in data and scaler.vol_scaler is not None:
            scaler.vol_scaler.global_std = data['vol_scaler']['global_std']
        
        return scaler


class ScalerGroup:
    """
    Group of scalers for different input types.
    Fits on training data only, then transforms all splits.
    """
    
    def __init__(self):
        self.scaler_short = StandardScaler()
        self.scaler_mid = StandardScaler()
        self.scaler_long = StandardScaler()
        self.scaler_context = StandardScaler()
        # Sentiment features also need scaling? They are roughly in range [-1, 1] for mean, but count can be large.
        # So yes, scaler for sentiment is good.
        self.scaler_sentiment = StandardScaler()
        self.fitted = False
    
    def fit(self, train_samples: List[Sample]) -> None:
        """
        Fit scalers on training samples only.
        
        Args:
            train_samples: List of training samples
        """
        logger.info(f"Fitting scalers on {len(train_samples)} training samples")
        
        # Collect all data
        short_data = []
        mid_data = []
        long_data = []
        context_data = []
        sentiment_data = []
        
        for sample in train_samples:
            # Flatten temporal dimensions for fitting
            short_data.append(sample.X_short.reshape(-1, sample.X_short.shape[-1]))
            mid_data.append(sample.X_mid.reshape(-1, sample.X_mid.shape[-1]))
            long_data.append(sample.X_long.reshape(-1, sample.X_long.shape[-1]))
            context_data.append(sample.X_context)
            if sample.X_sentiment.size > 0:
                sentiment_data.append(sample.X_sentiment)
        
        # Concatenate along sample dimension
        short_all = np.vstack(short_data)
        mid_all = np.vstack(mid_data)
        long_all = np.vstack(long_data)
        context_all = np.vstack(context_data)
        
        # Fit
        self.scaler_short.fit(short_all)
        self.scaler_mid.fit(mid_all)
        self.scaler_long.fit(long_all)
        self.scaler_context.fit(context_all)
        
        if sentiment_data:
            sentiment_all = np.vstack(sentiment_data)
            self.scaler_sentiment.fit(sentiment_all)
        
        self.fitted = True
        logger.info("Scalers fitted successfully")
    
    def transform(self, samples: List[Sample]) -> List[Sample]:
        """
        Transform samples using fitted scalers.
        
        Args:
            samples: List of samples to transform
        
        Returns:
            List of transformed samples (new objects)
        """
        if not self.fitted:
            raise RuntimeError("Scalers not fitted. Call fit() first.")
        
        transformed = []
        
        for sample in samples:
            # Transform each input type
            # Short: (win_short, n_feat) -> flatten -> scale -> reshape
            orig_short_shape = sample.X_short.shape
            X_short_flat = sample.X_short.reshape(-1, orig_short_shape[-1])
            X_short_scaled = self.scaler_short.transform(X_short_flat)
            X_short_scaled = X_short_scaled.reshape(orig_short_shape)
            
            # Mid
            orig_mid_shape = sample.X_mid.shape
            X_mid_flat = sample.X_mid.reshape(-1, orig_mid_shape[-1])
            X_mid_scaled = self.scaler_mid.transform(X_mid_flat)
            X_mid_scaled = X_mid_scaled.reshape(orig_mid_shape)
            
            # Long
            orig_long_shape = sample.X_long.shape
            X_long_flat = sample.X_long.reshape(-1, orig_long_shape[-1])
            X_long_scaled = self.scaler_long.transform(X_long_flat)
            X_long_scaled = X_long_scaled.reshape(orig_long_shape)
            
            # Context (1D)
            X_context_scaled = self.scaler_context.transform(
                sample.X_context.reshape(1, -1)
            ).flatten()
            
            # Sentiment
            if sample.X_sentiment.size > 0:
                X_sentiment_scaled = self.scaler_sentiment.transform(
                    sample.X_sentiment.reshape(1, -1)
                ).flatten()
            else:
                X_sentiment_scaled = sample.X_sentiment
            
            # Create new sample with scaled inputs
            transformed_sample = Sample(
                X_short=X_short_scaled.astype(np.float32),
                X_mid=X_mid_scaled.astype(np.float32),
                X_long=X_long_scaled.astype(np.float32),
                X_context=X_context_scaled.astype(np.float32),
                X_sentiment=X_sentiment_scaled.astype(np.float32),
                y=sample.y,  # Targets not scaled
                y_zone=sample.y_zone,
                y_levels=sample.y_levels,
                y_trend=sample.y_trend,
                date=sample.date,
                symbol=sample.symbol,
                stock_idx=sample.stock_idx,
                close_t=sample.close_t,
                vol_ref=sample.vol_ref
            )
            transformed.append(transformed_sample)
        
        return transformed
    
    def save(self, path: Path) -> None:
        """Save scalers to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler_short': self.scaler_short,
                'scaler_mid': self.scaler_mid,
                'scaler_long': self.scaler_long,
                'scaler_context': self.scaler_context,
                'scaler_sentiment': self.scaler_sentiment,
                'fitted': self.fitted
            }, f)
        
        logger.info(f"Scalers saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'ScalerGroup':
        """Load scalers from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        sg = ScalerGroup()
        sg.scaler_short = data['scaler_short']
        sg.scaler_mid = data['scaler_mid']
        sg.scaler_long = data['scaler_long']
        sg.scaler_context = data['scaler_context']
        sg.scaler_sentiment = data.get('scaler_sentiment', StandardScaler()) # Backward compat
        sg.fitted = data['fitted']
        
        logger.info(f"Scalers loaded from {path}")
        return sg
