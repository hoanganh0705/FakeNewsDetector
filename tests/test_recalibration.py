from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.post_hoc_calibration import (
    CalibratedProb,
    evaluate_recalibration,
    isotonic_regression,
    platt_scaling,
    temperature_scaling,
)
from src.evaluation.calibration_analysis import expected_calibration_error


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _well_calibrated_data(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(seed)
    logits = rng.normal(loc=0.0, scale=1.5, size=n)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return logits, y


def _miscalibrated_data(
    n: int = 4000,
    seed: int = 0,
    overconfidence: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = _rng(seed)
    well = rng.normal(loc=0.0, scale=1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-well))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    over = well * overconfidence
    return well, over, y


class TestTemperatureScaling:
    def test_perfect_calibration_recovers_t_one(self):
        logits, y = _well_calibrated_data(n=10000, seed=0)
        cal = temperature_scaling(logits, y, T_init=0.7)  # bad init to prove robustness
        T = cal.params["T"]
        assert math_close(T, 1.0, abs_tol=0.15)

    def test_t_is_positive_and_finite(self):
        logits, y = _well_calibrated_data(n=500)
        cal = temperature_scaling(logits, y)
        assert np.isfinite(cal.params["T"])
        assert cal.params["T"] > 0.0

    def test_predict_proba_returns_finite_probabilities(self):
        logits, y = _well_calibrated_data(n=500)
        cal = temperature_scaling(logits, y)
        probe = np.linspace(-30.0, 30.0, 100)
        p = cal(probe)
        assert np.all(np.isfinite(p))
        assert np.all((p >= 0.0) & (p <= 1.0))


class TestPlattScaling:
    def test_lowers_ece_on_overconfident_data(self):
        well, over, y = _miscalibrated_data(n=4000, seed=1, overconfidence=1.7)
        p_over = 1.0 / (1.0 + np.exp(-over))
        ece_before = expected_calibration_error(y, p_over, n_bins=10)

        cal = platt_scaling(over, y)
        p_after = cal(over)
        ece_after = expected_calibration_error(y, p_after, n_bins=10)

        assert ece_after < ece_before
        assert ece_after < ece_before / 2.0  # at least 50% reduction

    def test_params_are_finite(self):
        _, over, y = _miscalibrated_data(n=500)
        cal = platt_scaling(over, y)
        assert np.isfinite(cal.params["a"])
        assert np.isfinite(cal.params["b"])

    def test_predict_proba_in_unit_interval(self):
        _, over, y = _miscalibrated_data(n=500)
        cal = platt_scaling(over, y)
        probe = np.linspace(-20.0, 20.0, 100)
        p = cal(probe)
        assert np.all((p >= 0.0) & (p <= 1.0))


class TestIsotonicRegression:
    def test_lowers_ece_on_overconfident_data(self):
        _, over, y = _miscalibrated_data(n=4000, seed=2, overconfidence=1.7)
        p_over = 1.0 / (1.0 + np.exp(-over))
        ece_before = expected_calibration_error(y, p_over, n_bins=10)

        cal = isotonic_regression(over, y)
        p_after = cal(over)
        ece_after = expected_calibration_error(y, p_after, n_bins=10)
        assert ece_after <= ece_before

    def test_output_is_monotone_in_input(self):
        _, over, y = _miscalibrated_data(n=500)
        cal = isotonic_regression(over, y)
        probe = np.linspace(-10.0, 10.0, 200)
        p = cal(probe)
        assert np.all(np.diff(p) >= -1e-12)

    def test_output_stays_in_unit_interval(self):
        _, over, y = _miscalibrated_data(n=500)
        cal = isotonic_regression(over, y)
        probe = np.linspace(-20.0, 20.0, 100)
        p = cal(probe)
        assert np.all((p >= 0.0) & (p <= 1.0))


class TestMonotoneNonWorsening:

    @pytest.mark.parametrize("method", ["platt", "temperature", "isotonic"])
    def test_ece_non_increasing_after_recalibration(self, method):
        # Mis-calibrated data ⇒ meaningful improvement is possible.
        _, over, y = _miscalibrated_data(n=2000, seed=3, overconfidence=1.5)
        n_fit = int(0.8 * len(over))
        logits_fit, y_fit = over[:n_fit], y[:n_fit]
        logits_val, y_val = over[n_fit:], y[n_fit:]

        p_orig = 1.0 / (1.0 + np.exp(-logits_val))
        ece_orig = expected_calibration_error(y_val, p_orig, n_bins=10)

        if method == "platt":
            cal = platt_scaling(logits_fit, y_fit)
        elif method == "temperature":
            cal = temperature_scaling(logits_fit, y_fit)
        else:
            cal = isotonic_regression(logits_fit, y_fit)

        p_recal = cal(logits_val)
        ece_recal = expected_calibration_error(y_val, p_recal, n_bins=10)

        assert ece_recal <= ece_orig, (
            f"{method} made ECE worse: {ece_orig:.4f} -> {ece_recal:.4f}"
        )   


class TestEvaluateRecalibration:
    def test_returns_original_and_recalibrated_keys(self):
        logits, y = _well_calibrated_data(n=500)
        p = 1.0 / (1.0 + np.exp(-logits))
        cal = temperature_scaling(logits[:250], y[:250])
        p_recal = cal(logits)
        result = evaluate_recalibration(y, p, p_recal)
        assert set(result.keys()) == {"original", "recalibrated"}
        for side in ("original", "recalibrated"):
            for key in ("accuracy", "f1_macro", "roc_auc", "ece", "mce", "brier"):
                assert key in result[side], f"{side} missing {key}"
                assert np.isfinite(result[side][key])


def math_close(a: float, b: float, abs_tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= abs_tol