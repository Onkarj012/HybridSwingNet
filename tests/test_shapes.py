"""
Test tensor shapes throughout the pipeline.
"""

import pytest
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stockxpert.models.stockxpert import StockXpertModel
from stockxpert.dataset.make_samples import Sample


def test_sample_shapes():
    """Test that samples have correct shapes."""
    sample = Sample(
        X_short=np.random.randn(7, 5).astype(np.float32),
        X_mid=np.random.randn(21, 9).astype(np.float32),
        X_long=np.random.randn(60, 6).astype(np.float32),
        X_context=np.random.randn(10).astype(np.float32),
        X_sentiment=np.random.randn(9).astype(np.float32),
        y=np.zeros(5, dtype=np.float32),
        y_zone=np.zeros(5, dtype=np.int64),
        y_levels=np.zeros((1, 3), dtype=np.float32),
        y_trend=np.array([1.0], dtype=np.float32),
        date=pd.Timestamp("2023-01-01"),
        symbol='TEST.NS',
        stock_idx=0,
        close_t=100.0,
        vol_ref=1.0
    )
    
    assert sample.X_short.shape == (7, 5), f"Short shape mismatch: {sample.X_short.shape}"
    assert sample.X_mid.shape == (21, 9), f"Mid shape mismatch: {sample.X_mid.shape}"
    assert sample.X_long.shape == (60, 6), f"Long shape mismatch: {sample.X_long.shape}"
    assert sample.X_context.shape == (10,), f"Context shape mismatch: {sample.X_context.shape}"
    assert sample.X_sentiment.shape == (9,), f"Sentiment shape mismatch: {sample.X_sentiment.shape}"
    assert sample.y.shape == (5,), f"Target shape mismatch: {sample.y.shape}"
    assert sample.y_zone.shape == (5,), f"Zone shape mismatch: {sample.y_zone.shape}"
    
    print("✓ Sample shapes correct")


def test_model_output_shapes():
    """Test that model outputs have correct shapes."""
    batch_size = 16
    num_horizons = 5
    
    model = StockXpertModel(
        num_stocks=10,
        short_dim=5,
        mid_dim=9,
        long_dim=6,
        context_dim=10,
        num_horizons=num_horizons,
        hidden_dim=96,
        stock_embed_dim=16,
        attn_heads=4,
        dropout=0.15
    )
    
    # Create dummy inputs
    X_short = torch.randn(batch_size, 7, 5)
    X_mid = torch.randn(batch_size, 21, 9)
    X_long = torch.randn(batch_size, 60, 6)
    X_context = torch.randn(batch_size, 10)
    X_sentiment = torch.randn(batch_size, 9)
    stock_idx = torch.randint(0, 10, (batch_size,))

    # Forward pass
    direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, attention_dict = model(
        X_short, X_mid, X_long, X_context, X_sentiment, stock_idx
    )

    assert direction_logits.shape == (batch_size, num_horizons), \
        f"Direction logits shape mismatch: {direction_logits.shape}"
    assert magnitude_pred.shape == (batch_size, num_horizons), \
        f"Magnitude pred shape mismatch: {magnitude_pred.shape}"
    assert confidence.shape == (batch_size, num_horizons), \
        f"Confidence shape mismatch: {confidence.shape}"
    assert target_zone_logits.shape == (batch_size, num_horizons, 7), \
        f"Target zone logits shape mismatch: {target_zone_logits.shape}"
    
    print("✓ Model output shapes correct")


if __name__ == '__main__':
    test_sample_shapes()
    test_model_output_shapes()
    print("\nAll shape tests passed!")
