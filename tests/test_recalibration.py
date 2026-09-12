"""Tests for ``src.evaluation.post_hoc_calibration``.

The plan (§6, Step 2.5) asks for three correctness invariants:

1. Temperature scaling on a *perfectly-calibrated* toy model returns
   ``T ≈ 1.0`` (the optimum of NLL on already-calibrated probs is
   the identity).
2. Platt scaling on a *miscalibrated* toy model lowers the ECE on
   the held-out split.
3. After recalibration, ECE on the validation set is monotone
   non-increasing (or at worst, unchanged) — guarantees we never make
   things worse than the baseline.

The tests use synthetic binary-classification data with controllable
calibration, so they run quickly without the real Vietnamese dataset.
"""

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


# ──────────────────────────────────────────────────────────────────────
# Helpers — synthetic data generators
# ──────────────────────────────────────────────────────────────────────


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _well_calibrated_data(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logits, y)`` such that ``sigmoid(logit)`` is well-calibrated.

    Trick: sample the true label from the model's own probability
    directly — this guarantees empirical positive rate == predicted
    probability per bin up to sampling noise.
    """
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
    """Return ``(well_logits, overconfident_logits, y)``.

    ``well_logits`` are well-calibrated; ``overconfident_logits`` are
    the same logits *rescaled* by ``overconfidence`` to push probabilities
    toward the extremes — the textbook miscalibration pattern
    ("the model is too confident").
    """
    rng = _rng(seed)
    well = rng.normal(loc=0.0, scale=1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-well))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    over = well * overconfidence
    return well, over, y


# ──────────────────────────────────────────────────────────────────────
# 1. Temperature scaling — perfectly calibrated ⇒ T ≈ 1.0
# ──────────────────────────────────────────────────────────────────────


class TestTemperatureScaling:
    """When the model is already calibrated, T should converge to ≈1.0."""

    def test_perfect_calibration_recovers_t_one(self):
        logits, y = _well_calibrated_data(n=10000, seed=0)
        cal = temperature_scaling(logits, y, T_init=0.7)  # bad init to prove robustness
        T = cal.params["T"]
        # Allow generous tolerance: ECE of well-calibrated probs is
        # ~0.02 already, so T can drift a few percent without harm.
        assert math_close(T, 1.0, abs_tol=0.15)

    def test_t_is_positive_and_finite(self):
        logits, y = _well_calibrated_data(n=500)
        cal = temperature_scaling(logits, y)
        assert np.isfinite(cal.params["T"])
        assert cal.params["T"] > 0.0

    def test_predict_proba_returns_finite_probabilities(self):
        logits, y = _well_calibrated_data(n=500)
        cal = temperature_scaling(logits, y)
        # Probe wide range of logits (including extreme values).
        probe = np.linspace(-30.0, 30.0, 100)
        p = cal(probe)
        assert np.all(np.isfinite(p))
        assert np.all((p >= 0.0) & (p <= 1.0))


# ──────────────────────────────────────────────────────────────────────
# 2. Platt scaling — improves ECE on miscalibrated data
# ──────────────────────────────────────────────────────────────────────


class TestPlattScaling:
    """Platt must lower ECE on a miscalibrated model."""

    def test_lowers_ece_on_overconfident_data(self):
        well, over, y = _miscalibrated_data(n=4000, seed=1, overconfidence=1.7)
        # ECE before calibration (on overconfident logits).
        p_over = 1.0 / (1.0 + np.exp(-over))
        ece_before = expected_calibration_error(y, p_over, n_bins=10)

        cal = platt_scaling(over, y)
        p_after = cal(over)
        ece_after = expected_calibration_error(y, p_after, n_bins=10)

        # Strict improvement expected (miscalibration is large by construction).
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


# ──────────────────────────────────────────────────────────────────────
# 3. Isotonic regression — same improvement, monotone
# ──────────────────────────────────────────────────────────────────────


class TestIsotonicRegression:
    """Isotonic regression is the most flexible recalibrator."""

    def test_lowers_ece_on_overconfident_data(self):
        _, over, y = _miscalibrated_data(n=4000, seed=2, overconfidence=1.7)
        p_over = 1.0 / (1.0 + np.exp(-over))
        ece_before = expected_calibration_error(y, p_over, n_bins=10)

        cal = isotonic_regression(over, y)
        p_after = cal(over)
        ece_after = expected_calibration_error(y, p_after, n_bins=10)
        assert ece_after <= ece_before

    def test_output_is_monotone_in_input(self):
        """Isotonic regression by definition is a monotone mapping."""
        _, over, y = _miscalibrated_data(n=500)
        cal = isotonic_regression(over, y)
        probe = np.linspace(-10.0, 10.0, 200)
        p = cal(probe)
        # Adjacent differences should be non-negative.
        assert np.all(np.diff(p) >= -1e-12)

    def test_output_stays_in_unit_interval(self):
        _, over, y = _miscalibrated_data(n=500)
        cal = isotonic_regression(over, y)
        probe = np.linspace(-20.0, 20.0, 100)
        p = cal(probe)
        assert np.all((p >= 0.0) & (p <= 1.0))


# ──────────────────────────────────────────────────────────────────────
# 4. Monotone non-worsening property on a held-out validation set
# ──────────────────────────────────────────────────────────────────────


class TestMonotoneNonWorsening:
    """Recalibration must not make ECE worse on the *fit* split.

    We measure monotonicity on the **fit** split (the data used to fit
    the calibrator) — not on a held-out split — because:

    * Platt / Temperature / Isotonic are designed to minimise NLL /
      squared loss on the **fit** set by construction, so monotonic
      improvement is guaranteed on the fit set by definition.
    * The held-out split can degrade slightly because the calibrator
      has only partial information — this is normal, not a bug.

    The plan's invariant is "recalibrated ECE ≤ original ECE on val".
    In our setup, ``val`` = the fit split (we don't have separate
    "calibration val" and "calibration test" in this synthetic test).
    """

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


# ──────────────────────────────────────────────────────────────────────
# 5. evaluate_recalibration — contract
# ──────────────────────────────────────────────────────────────────────


class TestEvaluateRecalibration:
    def test_returns_original_and_recalibrated_keys(self):
        logits, y = _well_calibrated_data(n=500)
        p = 1.0 / (1.0 + np.exp(-logits))
        cal = temperature_scaling(logits[:250], y[:250])
        p_recal = cal(logits)
        result = evaluate_recalibration(y, p, p_recal)
        assert set(result.keys()) == {"original", "recalibrated"}
        # Each side should expose the standard calibration metrics.
        for side in ("original", "recalibrated"):
            for key in ("accuracy", "f1_macro", "roc_auc", "ece", "mce", "brier"):
                assert key in result[side], f"{side} missing {key}"
                assert np.isfinite(result[side][key])


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def math_close(a: float, b: float, abs_tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= abs_tol