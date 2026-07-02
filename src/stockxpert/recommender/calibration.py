"""
Calibration module for StockXpert predictions.
Uses Isotonic Regression to map raw model outputs to calibrated probabilities and returns.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from typing import Dict, List, Optional, Tuple

class PredictionCalibrator:
    """
    Calibrates model predictions using validation data.
    
    Capabilities:
    1. Calibrate Magnitude (Z-score -> Actual Return %)
    2. Calibrate Probability (Sigmoid -> Empirical Probability)
    """
    
    def __init__(self, horizons: List[int] = [1, 3, 5, 7, 10]):
        self.horizons = horizons
        self.mag_regressors = {}  # Map horizon -> IsotonicRegression
        self.prob_regressors = {} # Map horizon -> IsotonicRegression
        self.fitted = False
        
    def fit(self, 
            mag_preds: np.ndarray, 
            mag_targets: np.ndarray,
            prob_preds: np.ndarray,
            dir_targets: np.ndarray):
        """
        Fit calibrators on validation data.
        
        Args:
            mag_preds: (N, H) raw magnitude predictions (Z-scores)
            mag_targets: (N, H) actual target returns (log returns)
            prob_preds: (N, H) raw probability scores (0-1)
            dir_targets: (N, H) binary direction targets (0 or 1)
        """
        for i, h in enumerate(self.horizons):
            # 1. Magnitude Calibration (Z-score -> Return)
            # We use isotonic regression to map rank-order of z-scores to actual returns
            # But Isotonic requires varying monotonicity. Return vs Z-score should be monotonically increasing.
            
            # Filter valid data
            valid_mask = ~np.isnan(mag_preds[:, i]) & ~np.isnan(mag_targets[:, i])
            X_mag = mag_preds[valid_mask, i]
            y_mag = mag_targets[valid_mask, i]
            
            if len(X_mag) > 100:
                iso_mag = IsotonicRegression(out_of_bounds='clip', increasing=True)
                iso_mag.fit(X_mag, y_mag)
                self.mag_regressors[h] = iso_mag
            else:
                self.mag_regressors[h] = None
                
            # 2. Probability Calibration
            valid_mask_prob = ~np.isnan(prob_preds[:, i]) & ~np.isnan(dir_targets[:, i])
            X_prob = prob_preds[valid_mask_prob, i]
            y_dir = dir_targets[valid_mask_prob, i]
            
            if len(X_prob) > 100:
                iso_prob = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
                iso_prob.fit(X_prob, y_dir)
                self.prob_regressors[h] = iso_prob
            else:
                self.prob_regressors[h] = None
                
        self.fitted = True
        
    def calibrate(self, 
                  mag_preds: np.ndarray, 
                  prob_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate new predictions.
        
        Returns:
            (calibrated_returns, calibrated_probs)
        """
        if not self.fitted:
            # Fallback to simple scaling if not fitted
            return mag_preds * 0.01, prob_preds
            
        N, H = mag_preds.shape
        cal_returns = np.zeros_like(mag_preds)
        cal_probs = np.zeros_like(prob_preds)
        
        for i, h in enumerate(self.horizons):
            # Magnitude
            reg = self.mag_regressors.get(h)
            if reg:
                cal_returns[:, i] = reg.predict(mag_preds[:, i])
            else:
                cal_returns[:, i] = mag_preds[:, i] * 0.01 # Fallback
                
            # Probability
            reg_prob = self.prob_regressors.get(h)
            if reg_prob:
                cal_probs[:, i] = reg_prob.predict(prob_preds[:, i])
            else:
                cal_probs[:, i] = prob_preds[:, i]
                
        return cal_returns, cal_probs
        
    def save(self, path: str):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> 'PredictionCalibrator':
        return joblib.load(path)
