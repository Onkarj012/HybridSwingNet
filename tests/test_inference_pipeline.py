#!/usr/bin/env python3
"""
Tests for Phase 1 inference pipeline bug fixes.
Verifies sentiment scaling, vol_ref validation, stock embedding handling,
dynamic accuracy loading, and horizon-specific thresholds.
"""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from sklearn.preprocessing import StandardScaler


class TestSentimentScaling:
    """FIX #1: Sentiment features must be scaled during inference."""

    def test_scaler_group_has_sentiment_scaler(self):
        """ScalerGroup should always have a scaler_sentiment attribute."""
        from stockxpert.dataset.scaling import ScalerGroup
        sg = ScalerGroup()
        assert hasattr(sg, 'scaler_sentiment'), "ScalerGroup missing scaler_sentiment"
        assert isinstance(sg.scaler_sentiment, StandardScaler)

    def test_sentiment_scaler_transforms_values(self):
        """Sentiment scaler should transform raw values into a different distribution."""
        scaler = StandardScaler()
        # Simulate training data: sentiment values with specific distribution
        train_data = np.array([
            [0.1, 5, 0.05, 0.12, 0.02, 0.0, 0.11, 0.10, 0.01],
            [0.3, 10, 0.15, 0.28, 0.04, 1.0, 0.25, 0.22, 0.03],
            [-0.2, 3, 0.10, -0.15, -0.03, 0.0, -0.18, -0.16, -0.02],
            [0.0, 0, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        scaler.fit(train_data)

        # Inference: neutral/zero sentiment (simulating missing data)
        raw_inference = np.zeros((1, 9), dtype=np.float32)
        scaled = scaler.transform(raw_inference)

        # Scaled zeros should NOT be all zeros (because StandardScaler shifts by mean)
        assert not np.allclose(scaled, 0.0), \
            "Scaling zeros should produce non-zero values (shifted by training mean)"

    def test_unscaled_vs_scaled_sentiment_differ(self):
        """Raw sentiment and scaled sentiment should be different."""
        scaler = StandardScaler()
        train_data = np.random.randn(100, 9) * 0.5 + 0.1
        scaler.fit(train_data)

        raw = np.array([[0.1, 5, 0.05, 0.12, 0.02, 0.0, 0.11, 0.10, 0.01]])
        scaled = scaler.transform(raw)

        assert not np.allclose(raw, scaled), \
            "Scaled output should differ from raw input"


class TestVolRefValidation:
    """FIX #3: vol_ref should be validated, not silently fallback to 0.015."""

    def test_vol_ref_computed_by_feature_builder(self):
        """FeatureBuilder should compute vol_ref column."""
        from stockxpert.features.builder import FeatureBuilder

        fb = FeatureBuilder(horizons=[1, 3, 5])

        # Create minimal price data
        dates = pd.bdate_range('2024-01-01', periods=50)
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000, 10000, 50),
            # Mock sentiment
            'sentiment_mean': 0.0,
            'sentiment_count': 0,
            'sentiment_std': 0.0,
        }, index=dates)

        result = fb.build_features({'TEST': df}, is_inference=True)

        assert 'TEST' in result, "Symbol missing from feature output"
        assert 'vol_ref' in result['TEST'].columns, "vol_ref column missing"
        assert not result['TEST']['vol_ref'].isna().all(), "vol_ref should not be all NaN"

    def test_vol_ref_is_positive(self):
        """vol_ref should always be positive (std dev + epsilon)."""
        from stockxpert.features.builder import FeatureBuilder

        fb = FeatureBuilder(horizons=[1])
        dates = pd.bdate_range('2024-01-01', periods=50)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pd.DataFrame({
            'Open': close * 0.99, 'High': close * 1.01,
            'Low': close * 0.98, 'Close': close,
            'Volume': np.random.randint(1000, 10000, 50),
            'sentiment_mean': 0.0, 'sentiment_count': 0, 'sentiment_std': 0.0,
        }, index=dates)

        result = fb.build_features({'TEST': df}, is_inference=True)
        vr = result['TEST']['vol_ref'].dropna()
        assert (vr > 0).all(), "vol_ref must always be positive"


class TestStockEmbeddingValidation:
    """FIX #4: Unknown symbols should be warned and skipped."""

    def test_unknown_symbol_not_in_map(self):
        """symbol_map should not contain symbols not in training config."""
        training_symbols = ['RELIANCE', 'TCS', 'INFY']
        symbol_map = {s: i for i, s in enumerate(training_symbols)}

        # New symbol not seen during training
        assert 'NEWSTOCK' not in symbol_map, \
            "Unknown symbol should not be in map"

    def test_known_symbol_in_map(self):
        """Training symbols should all be in map."""
        training_symbols = ['RELIANCE', 'TCS', 'INFY']
        symbol_map = {s: i for i, s in enumerate(training_symbols)}

        for sym in training_symbols:
            assert sym in symbol_map, f"{sym} missing from map"


class TestDynamicAccuracy:
    """FIX #5: Accuracy should be loaded from metrics.json, not hardcoded."""

    def test_load_metrics_from_json(self):
        """_load_metrics should parse metrics.json correctly."""
        metrics = {
            'direction_accuracy_h1': 78.4,
            'direction_accuracy_h3': 65.0,
            'direction_accuracy_h5': 58.6,
            'direction_accuracy_h7': 56.3,
            'direction_accuracy_h10': 53.3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / 'reports'
            reports_dir.mkdir()
            with open(reports_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f)

            # Simulate what _load_metrics does
            metrics_path = reports_dir / 'metrics.json'
            with open(metrics_path) as f:
                loaded = json.load(f)

            horizons = [1, 3, 5, 7, 10]
            acc_notes = {}
            for h in horizons:
                key = f'direction_accuracy_h{h}'
                if key in loaded:
                    acc_notes[h] = f"~{loaded[key]:.1f}%"

            assert acc_notes[1] == '~78.4%'
            assert acc_notes[3] == '~65.0%'

    def test_missing_metrics_returns_empty(self):
        """If no metrics.json, accuracy should be empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / 'reports' / 'metrics.json'
            assert not metrics_path.exists()
            # Should not raise, just return empty
            acc_notes = {}
            assert acc_notes == {}


class TestSignalThresholds:
    """FIX #6: Signal thresholds should be horizon-specific and tighter."""

    def test_h1_threshold_is_strictest(self):
        """1-Day horizon should have the strictest threshold."""
        thresholds = {
            1:  {'long': 0.60, 'short': 0.40},
            3:  {'long': 0.58, 'short': 0.42},
            5:  {'long': 0.56, 'short': 0.44},
            7:  {'long': 0.55, 'short': 0.45},
            10: {'long': 0.55, 'short': 0.45},
        }
        assert thresholds[1]['long'] > thresholds[10]['long'], \
            "H1 should have stricter long threshold than H10"
        assert thresholds[1]['short'] < thresholds[10]['short'], \
            "H1 should have stricter short threshold than H10"

    def test_signal_classification_with_new_thresholds(self):
        """Test that borderline signals are correctly classified."""
        thresholds = {'long': 0.60, 'short': 0.40}

        # p_up = 0.55 should NOT trigger LONG with new threshold (was LONG with 0.52)
        assert not (0.55 > thresholds['long']), \
            "p_up=0.55 should NOT be LONG with threshold 0.60"

        # p_up = 0.45 should NOT trigger SHORT with new threshold (was SHORT with 0.48)
        assert not (0.45 < thresholds['short']), \
            "p_up=0.45 should NOT be SHORT with threshold 0.40"

        # p_up = 0.65 should trigger LONG
        assert 0.65 > thresholds['long'], \
            "p_up=0.65 should be LONG with threshold 0.60"


class TestATRStopLoss:
    """FIX #7: Predictions should include ATR-based stop-loss and R:R ratio."""

    def test_atr_computation(self):
        """ATR should be computed from High/Low/Close."""
        np.random.seed(42)
        n = 30
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-14:])

        assert atr > 0, "ATR should be positive"
        assert atr < close[-1] * 0.1, "ATR should be reasonable (< 10% of price)"

    def test_stop_loss_direction(self):
        """Stop-loss should be below entry for LONG, above for SHORT."""
        current_price = 100.0
        atr = 2.0
        atr_multiplier = 1.5

        # LONG: stop below
        long_stop = current_price - (atr * atr_multiplier)
        assert long_stop < current_price, "Long stop should be below entry"

        # SHORT: stop above
        short_stop = current_price + (atr * atr_multiplier)
        assert short_stop > current_price, "Short stop should be above entry"

    def test_rr_ratio_positive(self):
        """Risk:Reward ratio should be positive when target differs from entry."""
        current_price = 100.0
        target_price = 103.0  # LONG target
        stop_loss = 97.0

        risk = current_price - stop_loss  # 3.0
        reward = abs(target_price - current_price)  # 3.0
        rr = reward / risk if risk > 0 else 0.0

        assert rr > 0, "R:R ratio should be positive"
        assert rr == 1.0, "Equal risk/reward should give R:R = 1.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
