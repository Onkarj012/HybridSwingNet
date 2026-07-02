"""V3 paper-trading drift checks."""

from __future__ import annotations

import numpy as np

from stockxpert.v3.models.calibration import CalibrationEvaluator


def population_stability_index(expected, actual, n_bins: int = 10) -> float:
    expected = np.asarray(expected).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    quantiles = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, 1.0)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, 1.0)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


class DriftMonitor:
    def __init__(self, psi_threshold: float = 0.20, ece_drift_threshold: float = 0.05, accuracy_floor: float = 0.52):
        self.psi_threshold = psi_threshold
        self.ece_drift_threshold = ece_drift_threshold
        self.accuracy_floor = accuracy_floor
        self.calibration = CalibrationEvaluator()

    def feature_drift(self, train_features: dict[str, np.ndarray], live_features: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            name: population_stability_index(train_values, live_features[name])
            for name, train_values in train_features.items()
            if name in live_features
        }

    def prediction_drift(self, reference_probs, live_probs) -> float:
        try:
            from scipy.stats import ks_2samp

            return float(ks_2samp(np.asarray(reference_probs).reshape(-1), np.asarray(live_probs).reshape(-1)).pvalue)
        except Exception:
            return float("nan")

    def calibration_drift(self, live_probs, live_labels, val_ece: float) -> float:
        return self.calibration.ece(live_probs, live_labels) - val_ece

    def retraining_triggers(
        self,
        rolling_accuracy: float,
        psi_by_feature: dict[str, float],
        ece_drift: float,
    ) -> list[str]:
        triggers = []
        if rolling_accuracy < self.accuracy_floor:
            triggers.append("rolling_accuracy_below_floor")
        if any(value > self.psi_threshold for value in psi_by_feature.values()):
            triggers.append("feature_psi_above_threshold")
        if abs(ece_drift) > self.ece_drift_threshold:
            triggers.append("calibration_ece_drift")
        return triggers

