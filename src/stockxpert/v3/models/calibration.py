"""Post-hoc calibration utilities for V3."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class IsotonicCalibrator:
    """Fit isotonic regression on validation probabilities and labels."""

    def __init__(self):
        self.model = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(np.asarray(probs).reshape(-1), np.asarray(labels).reshape(-1))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        shape = np.asarray(probs).shape
        return self.model.predict(np.asarray(probs).reshape(-1)).reshape(shape)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        with Path(path).open("rb") as f:
            return pickle.load(f)


class TemperatureScaling:
    """Single-temperature scaling for binary direction logits."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        import torch
        import torch.nn.functional as F

        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.float32)
        log_temp = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits_t / log_temp.exp().clamp_min(1e-3), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temp.detach().exp().clamp_min(1e-3))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float32) / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaling":
        with Path(path).open("rb") as f:
            return pickle.load(f)


class CalibrationEvaluator:
    """Compute calibration metrics for binary probabilities."""

    def reliability_data(self, probs, labels, n_bins: int = 15) -> dict:
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(probs, bins[1:-1], right=True)

        bin_means, bin_accs, bin_counts = [], [], []
        for i in range(n_bins):
            mask = bin_ids == i
            count = int(mask.sum())
            bin_counts.append(count)
            bin_means.append(float(probs[mask].mean()) if count else 0.0)
            bin_accs.append(float(labels[mask].mean()) if count else 0.0)

        return {"bin_means": bin_means, "bin_accs": bin_accs, "bin_counts": bin_counts}

    def ece(self, probs, labels, n_bins: int = 15) -> float:
        data = self.reliability_data(probs, labels, n_bins)
        counts = np.asarray(data["bin_counts"], dtype=np.float64)
        if counts.sum() == 0:
            return 0.0
        gaps = np.abs(np.asarray(data["bin_means"]) - np.asarray(data["bin_accs"]))
        return float((gaps * counts).sum() / counts.sum())

    def mce(self, probs, labels, n_bins: int = 15) -> float:
        data = self.reliability_data(probs, labels, n_bins)
        counts = np.asarray(data["bin_counts"])
        gaps = np.abs(np.asarray(data["bin_means"]) - np.asarray(data["bin_accs"]))
        return float(gaps[counts > 0].max()) if np.any(counts > 0) else 0.0

    def brier_score(self, probs, labels) -> float:
        probs = np.asarray(probs).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        return float(np.mean((probs - labels) ** 2))

