"""Tests for src.analysis.statistical_tests."""

import numpy as np
import pytest

from src.analysis.statistical_tests import mcnemar_test, holm_bonferroni_correction


class TestMcNemarTest:
    """Tests for the McNemar's test implementation."""

    def test_identical_predictions_no_difference(self):
        """Two identical classifiers → p-value = 1.0."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])  # same for both
        chi2, p = mcnemar_test(y_true, y_pred, y_pred.copy())
        assert p == pytest.approx(1.0)

    def test_completely_different_predictions(self):
        """One model always right, other always wrong → very small p."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # perfect
        y_pred2 = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # all wrong
        chi2, p = mcnemar_test(y_true, y_pred1, y_pred2)
        assert p < 0.05

    def test_returns_tuple_of_two_floats(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 0])
        y_pred2 = np.array([0, 0, 0, 1])
        result = mcnemar_test(y_true, y_pred1, y_pred2)
        assert len(result) == 2
        chi2, p = result
        assert isinstance(chi2, float)
        assert 0.0 <= p <= 1.0


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni correction."""

    def test_output_length_matches_input(self):
        p_values = [0.01, 0.04, 0.20]
        results = holm_bonferroni_correction(p_values)
        assert len(results) == len(p_values)

    def test_very_small_p_stays_significant(self):
        """A very small p-value should remain significant after correction."""
        p_values = [0.0001, 0.5, 0.9]
        results = holm_bonferroni_correction(p_values)
        # index 0 has p=0.0001, should still be significant
        assert results[0]["significant"] is True

    def test_large_p_not_significant(self):
        """A large p-value should not be significant."""
        p_values = [0.4, 0.6, 0.9]
        results = holm_bonferroni_correction(p_values)
        for r in results:
            assert r["significant"] is False

    def test_adjusted_p_is_non_decreasing(self):
        """Adjusted p-values sorted by raw p should be non-decreasing."""
        p_values = [0.01, 0.02, 0.04, 0.10]
        results = holm_bonferroni_correction(p_values)
        adjusted = sorted(zip(p_values, [r["adjusted_p"] for r in results]))
        adj_only = [a for _, a in adjusted]
        for i in range(1, len(adj_only)):
            assert adj_only[i] >= adj_only[i - 1]
