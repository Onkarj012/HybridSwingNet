"""Horizon/regime calibration with minimum-sample fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from stockxpert.v3.models.calibration import CalibrationEvaluator, IsotonicCalibrator, TemperatureScaling


@dataclass
class CalibrationRecord:
    horizon: int
    regime: str
    method: str
    n_samples: int
    ece: float


class HorizonRegimeCalibrator:
    def __init__(self, method: str = "isotonic", min_samples: int = 100):
        self.method = method
        self.min_samples = min_samples
        self.models: dict[tuple[int, str], object] = {}
        self.global_models: dict[int, object] = {}
        self.records: list[CalibrationRecord] = []

    def _make(self):
        if self.method == "temperature":
            return TemperatureScaling()
        return IsotonicCalibrator()

    def _fit_one(self, probs_or_logits: np.ndarray, labels: np.ndarray):
        model = self._make()
        return model.fit(probs_or_logits, labels)

    def fit(self, frame: pd.DataFrame, validation_fold_ids: Iterable[object] | None = None) -> "HorizonRegimeCalibrator":
        data = frame.copy()
        if validation_fold_ids is not None and "fold_id" in data.columns:
            data = data[data["fold_id"].isin(set(validation_fold_ids))]
        required = {"horizon", "regime", "p_up", "label"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing calibration columns: {sorted(missing)}")

        evaluator = CalibrationEvaluator()
        for horizon, h_df in data.groupby("horizon"):
            self.global_models[int(horizon)] = self._fit_one(h_df["p_up"].to_numpy(), h_df["label"].to_numpy())
            for regime, r_df in h_df.groupby("regime"):
                key = (int(horizon), str(regime))
                if len(r_df) >= self.min_samples:
                    model = self._fit_one(r_df["p_up"].to_numpy(), r_df["label"].to_numpy())
                    method = self.method
                    eval_probs = model.transform(r_df["p_up"].to_numpy())
                else:
                    model = self.global_models[int(horizon)]
                    method = "global_fallback"
                    eval_probs = model.transform(r_df["p_up"].to_numpy())
                self.models[key] = model
                self.records.append(
                    CalibrationRecord(
                        horizon=key[0],
                        regime=key[1],
                        method=method,
                        n_samples=int(len(r_df)),
                        ece=evaluator.ece(eval_probs, r_df["label"].to_numpy()),
                    )
                )
        return self

    def transform(self, horizon: int, regime: str, probs: np.ndarray) -> np.ndarray:
        model = self.models.get((int(horizon), str(regime))) or self.global_models.get(int(horizon))
        if model is None:
            return np.asarray(probs)
        return model.transform(probs)

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])
