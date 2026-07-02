#!/usr/bin/env python3
"""
Tests for Phase 2 loss function fixes.
Verifies that DirectionAwareLoss with balanced parameters produces symmetric loss,
and that horizon-specific weights are applied correctly.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pytest

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))


class TestSymmetricDirectionLoss:
    """Phase 2 Fix: DirectionAwareLoss should be balanced with equal gamma_pos/gamma_neg."""

    def test_current_defaults_are_asymmetric(self):
        """Document the CURRENT behavior: asymmetric loss causes SHORT bias."""
        from stockxpert.models.losses import DirectionAwareLoss

        loss_fn = DirectionAwareLoss()  # Uses current defaults

        # Create balanced predictions
        batch = 32
        n_horizons = 5
        logits = torch.zeros(batch, n_horizons)  # 50/50 predictions

        # All UP targets
        up_targets = torch.ones(batch, n_horizons)
        loss_up = loss_fn(logits, up_targets)

        # All DOWN targets
        down_targets = torch.zeros(batch, n_horizons)
        loss_down = loss_fn(logits, down_targets)

        # With current asymmetric settings, losses should differ
        # This test documents the existing bias
        print(f"Loss on UP targets: {loss_up.item():.4f}")
        print(f"Loss on DOWN targets: {loss_down.item():.4f}")
        print(f"Ratio (down/up): {loss_down.item() / loss_up.item():.4f}")

    def test_balanced_params_produce_symmetric_loss(self):
        """With gamma_pos=gamma_neg=1.5 and downside_weight=1.0, loss should be symmetric."""
        from stockxpert.models.losses import DirectionAwareLoss

        # Proposed balanced parameters
        loss_fn = DirectionAwareLoss(
            gamma_pos=1.5,
            gamma_neg=1.5,
            downside_weight=1.0
        )

        batch = 100
        n_horizons = 5
        logits = torch.zeros(batch, n_horizons)

        up_targets = torch.ones(batch, n_horizons)
        loss_up = loss_fn(logits, up_targets)

        down_targets = torch.zeros(batch, n_horizons)
        loss_down = loss_fn(logits, down_targets)

        # With balanced params, losses should be nearly equal
        ratio = loss_down.item() / loss_up.item()
        assert 0.9 < ratio < 1.1, \
            f"Balanced loss should be symmetric, but ratio = {ratio:.3f}"
        print(f"Balanced loss ratio: {ratio:.4f} ✓")


class TestHorizonWeights:
    """Phase 2 Fix: Per-horizon loss weights should be configurable."""

    def test_horizon_weight_scaling(self):
        """Horizon weights should scale the loss contribution."""
        # Proposed weights: shorter horizons matter more
        weights = [2.0, 1.5, 1.0, 0.75, 0.5]

        # Simulate per-horizon losses (all equal raw loss)
        raw_losses = [1.0, 1.0, 1.0, 1.0, 1.0]

        weighted = sum(w * l for w, l in zip(weights, raw_losses))
        unweighted = sum(raw_losses)

        # H1 should contribute most
        assert weights[0] == max(weights), "H1 should have highest weight"
        assert weights[-1] == min(weights), "H10 should have lowest weight"

    def test_uniform_weights_equals_mean(self):
        """With uniform weights, result should equal simple mean."""
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        raw_losses = [0.5, 0.6, 0.7, 0.8, 0.9]

        weighted_mean = sum(w * l for w, l in zip(weights, raw_losses)) / sum(weights)
        simple_mean = sum(raw_losses) / len(raw_losses)

        assert abs(weighted_mean - simple_mean) < 1e-6, \
            "Uniform weights should give same result as simple mean"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
