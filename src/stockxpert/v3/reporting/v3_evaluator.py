"""V3 statistical evaluator and cross-sectional breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from stockxpert.v3.models.calibration import CalibrationEvaluator


@dataclass
class EvalResults:
    metrics: dict[str, float] = field(default_factory=dict)
    statistical_tests: dict[str, float] = field(default_factory=dict)


@dataclass
class BreakdownResults:
    groups: dict[str, dict[str, float]] = field(default_factory=dict)


class V3Evaluator:
    def __init__(self, n_bootstrap: int = 1000, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.calibration = CalibrationEvaluator()

    def evaluate(self, predictions, labels, metadata=None) -> EvalResults:
        probs = np.asarray(predictions)
        y = np.asarray(labels).astype(int)
        pred_labels = (probs >= 0.5).astype(int)
        correct = pred_labels == y

        metrics = {
            "accuracy": float(correct.mean()),
            "precision": self._precision(pred_labels, y),
            "recall": self._recall(pred_labels, y),
            "f1": self._f1(pred_labels, y),
            "brier": self.calibration.brier_score(probs, y),
            "ece": self.calibration.ece(probs, y),
            "mce": self.calibration.mce(probs, y),
        }
        lo, hi = self.bootstrap_ci(lambda idx: correct.reshape(-1)[idx].mean(), len(correct.reshape(-1)))
        tests = {
            "binomial_p_value": self.binomial_test(int(correct.sum()), int(correct.size), p=0.5),
            "accuracy_ci_low": lo,
            "accuracy_ci_high": hi,
        }
        return EvalResults(metrics=metrics, statistical_tests=tests)

    def breakdown(self, predictions, labels, metadata) -> BreakdownResults:
        probs = np.asarray(predictions).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        correct = ((probs >= 0.5).astype(int) == y.astype(int)).astype(float)
        groups: dict[str, dict[str, float]] = {}

        for key in [
            "sector",
            "market_cap_bucket",
            "volatility_regime",
            "liquidity_bucket",
            "market_period",
            "sample_type",
        ]:
            if metadata is None or key not in metadata:
                continue
            values = np.asarray(metadata[key]).reshape(-1)
            for value in np.unique(values):
                mask = values == value
                groups[f"{key}:{value}"] = {
                    "n": int(mask.sum()),
                    "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
                }

        return BreakdownResults(groups=groups)

    def binomial_test(self, n_correct: int, n_total: int, p: float = 0.5) -> float:
        try:
            from scipy.stats import binomtest

            return float(binomtest(n_correct, n_total, p=p, alternative="greater").pvalue)
        except Exception:
            return float("nan")

    def bootstrap_ci(
        self,
        metric_fn: Callable[[np.ndarray], float],
        n: int,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        if n == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(self.seed)
        values = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, n)
            values.append(metric_fn(idx))
        alpha = (1.0 - ci) / 2.0
        return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))

    def mcnemar_test(self, model_correct, baseline_correct) -> float:
        model_correct = np.asarray(model_correct).astype(bool)
        baseline_correct = np.asarray(baseline_correct).astype(bool)
        b = int(np.logical_and(model_correct, ~baseline_correct).sum())
        c = int(np.logical_and(~model_correct, baseline_correct).sum())
        try:
            from scipy.stats import chi2

            statistic = (abs(b - c) - 1) ** 2 / max(b + c, 1)
            return float(1.0 - chi2.cdf(statistic, 1))
        except Exception:
            return float("nan")

    def _precision(self, pred, y) -> float:
        tp = np.logical_and(pred == 1, y == 1).sum()
        fp = np.logical_and(pred == 1, y == 0).sum()
        return float(tp / max(tp + fp, 1))

    def _recall(self, pred, y) -> float:
        tp = np.logical_and(pred == 1, y == 1).sum()
        fn = np.logical_and(pred == 0, y == 1).sum()
        return float(tp / max(tp + fn, 1))

    def _f1(self, pred, y) -> float:
        precision = self._precision(pred, y)
        recall = self._recall(pred, y)
        return float(2 * precision * recall / max(precision + recall, 1e-12))

