"""
General utility functions for StockXpert.
"""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Union, Dict, Optional, List
import time
from dataclasses import dataclass


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get PyTorch device.

    Args:
        device_str: One of 'cuda', 'mps', 'cpu', or 'auto'

    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str == "cuda" and not torch.cuda.is_available():
        if _mps_available():
            print("CUDA requested but not available, falling back to MPS")
            return torch.device("mps")
        print("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    if device_str == "mps":
        if _mps_available():
            return torch.device("mps")
        print("MPS requested but not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device(device_str)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"[{self.name}] Elapsed: {self.elapsed:.2f}s")


@dataclass
class AdaptiveThresholdState:
    """State for adaptive threshold computation."""
    recent_accuracy: float = 0.5
    recent_sharpe: float = 0.0
    recent_drawdown: float = 0.0
    market_volatility: float = 0.02
    n_recent_trades: int = 0


def adaptive_threshold(
    confidence: float,
    p_up: float,
    state: AdaptiveThresholdState,
    base_threshold: float = 0.5,
    min_threshold: float = 0.45,
    max_threshold: float = 0.65
) -> float:
    """
    Compute adaptive decision threshold based on multiple factors.
    
    The threshold adjusts based on:
    1. Recent prediction accuracy (higher accuracy → lower threshold)
    2. Recent Sharpe ratio (good risk-adjusted returns → lower threshold)
    3. Current drawdown (in drawdown → higher threshold)
    4. Market volatility (high vol → higher threshold)
    5. Model confidence (high confidence → can use lower threshold)
    
    Args:
        confidence: Model's confidence score (0-1)
        p_up: Model's probability of up move (0-1)
        state: AdaptiveThresholdState with recent performance metrics
        base_threshold: Starting threshold value
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
    
    Returns:
        Adjusted threshold value
    """
    threshold = base_threshold
    adjustments = []
    
    # 1. Accuracy adjustment
    # If recent accuracy > 55%, we can lower threshold (trade more)
    # If recent accuracy < 50%, raise threshold (be more selective)
    if state.n_recent_trades >= 10:  # Only adjust if we have enough data
        accuracy_adj = (0.525 - state.recent_accuracy) * 0.4
        adjustments.append(('accuracy', accuracy_adj))
    
    # 2. Sharpe ratio adjustment
    # Good Sharpe (>1) → lower threshold, poor Sharpe (<0) → higher threshold
    if state.n_recent_trades >= 10:
        sharpe_adj = -0.05 * min(2, max(-2, state.recent_sharpe))
        adjustments.append(('sharpe', sharpe_adj))
    
    # 3. Drawdown adjustment
    # In drawdown → raise threshold to be more conservative
    if state.recent_drawdown < -0.05:
        dd_adj = abs(state.recent_drawdown) * 0.5
        adjustments.append(('drawdown', dd_adj))
    
    # 4. Volatility adjustment
    # High market volatility → raise threshold
    if state.market_volatility > 0.025:
        vol_adj = (state.market_volatility - 0.02) * 2
        adjustments.append(('volatility', vol_adj))
    
    # 5. Confidence-based adjustment
    # When model is very confident, we can use a lower threshold
    # When model is uncertain, use higher threshold
    if confidence > 0.8:
        conf_adj = -0.03
    elif confidence > 0.7:
        conf_adj = -0.015
    elif confidence < 0.55:
        conf_adj = 0.02
    else:
        conf_adj = 0
    adjustments.append(('confidence', conf_adj))
    
    # Apply all adjustments
    total_adj = sum(adj for _, adj in adjustments)
    threshold = base_threshold + total_adj
    
    # Clamp to bounds
    threshold = max(min_threshold, min(max_threshold, threshold))
    
    return threshold


def adaptive_threshold_vectorized(
    confidences: np.ndarray,
    p_ups: np.ndarray,
    state: AdaptiveThresholdState,
    base_threshold: float = 0.5,
    min_threshold: float = 0.45,
    max_threshold: float = 0.65
) -> np.ndarray:
    """
    Vectorized adaptive threshold computation for batch processing.
    
    Args:
        confidences: Array of confidence scores (N,)
        p_ups: Array of P(up) probabilities (N,)
        state: AdaptiveThresholdState with recent performance metrics
        base_threshold: Starting threshold value
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
    
    Returns:
        Array of adjusted thresholds (N,)
    """
    # Start with base threshold
    thresholds = np.full(len(confidences), base_threshold)
    
    # Compute base adjustment from state (same for all)
    state_adj = 0.0
    if state.n_recent_trades >= 10:
        state_adj += (0.525 - state.recent_accuracy) * 0.4  # Accuracy
        state_adj += -0.05 * min(2, max(-2, state.recent_sharpe))  # Sharpe
    
    # Drawdown adjustment
    if state.recent_drawdown < -0.05:
        state_adj += abs(state.recent_drawdown) * 0.5
    
    # Volatility adjustment
    if state.market_volatility > 0.025:
        state_adj += (state.market_volatility - 0.02) * 2
    
    thresholds += state_adj
    
    # Per-sample confidence adjustment
    conf_adj = np.zeros_like(confidences)
    conf_adj[confidences > 0.8] = -0.03
    conf_adj[(confidences > 0.7) & (confidences <= 0.8)] = -0.015
    conf_adj[confidences < 0.55] = 0.02
    
    thresholds += conf_adj
    
    # Clamp
    thresholds = np.clip(thresholds, min_threshold, max_threshold)
    
    return thresholds


def should_trade(
    p_up: float,
    confidence: float,
    state: AdaptiveThresholdState,
    direction: str = 'long',
    base_threshold: float = 0.5
) -> tuple:
    """
    Determine if a trade should be taken based on adaptive threshold.
    
    Args:
        p_up: Model's probability of up move
        confidence: Model's confidence score
        state: Current adaptive threshold state
        direction: 'long' or 'short'
        base_threshold: Base probability threshold
    
    Returns:
        Tuple of (should_trade: bool, threshold_used: float)
    """
    threshold = adaptive_threshold(
        confidence=confidence,
        p_up=p_up,
        state=state,
        base_threshold=base_threshold
    )
    
    if direction == 'long':
        should = p_up > threshold
    else:  # short
        should = p_up < (1 - threshold)
    
    return should, threshold
