import numpy as np

from stockxpert.v3.models.calibration import CalibrationEvaluator, IsotonicCalibrator, TemperatureScaling


def test_isotonic_calibrator_roundtrip(tmp_path):
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    calibrator = IsotonicCalibrator().fit(probs, labels)
    path = tmp_path / "iso.pkl"
    calibrator.save(path)

    loaded = IsotonicCalibrator.load(path)
    assert np.allclose(calibrator.transform(probs), loaded.transform(probs))


def test_temperature_scaling_transforms_logits():
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    calibrator = TemperatureScaling().fit(logits, labels)
    probs = calibrator.transform(logits)

    assert probs.shape == logits.shape
    assert np.all((0.0 <= probs) & (probs <= 1.0))


def test_calibration_evaluator_known_values():
    evaluator = CalibrationEvaluator()
    probs = np.array([0.0, 1.0])
    labels = np.array([0, 1])

    assert evaluator.brier_score(probs, labels) == 0.0
    assert evaluator.ece(probs, labels, n_bins=2) == 0.0

