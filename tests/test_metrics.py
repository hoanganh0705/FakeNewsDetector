"""Tests for src.evaluation.metrics.compute_metrics()."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


class TestComputeMetrics:
    """Tests for the central compute_metrics() function."""

    def test_perfect_predictions(self):
        """All correct predictions → accuracy = 1.0, F1 = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        """All incorrect predictions → accuracy = 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_metrics_keys_present(self):
        """Result must contain the expected keys."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        m = compute_metrics(y_true, y_pred)
        required = {
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "confusion_matrix", "classification_report",
            "precision_per_class", "recall_per_class", "f1_per_class",
        }
        assert required.issubset(set(m.keys()))

    def test_roc_auc_computed_when_probs_given(self):
        """ROC-AUC key must be present only when y_prob is supplied."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])

        m_no_prob = compute_metrics(y_true, y_pred)
        m_with_prob = compute_metrics(y_true, y_pred, y_prob)

        assert "roc_auc" not in m_no_prob
        assert "roc_auc" in m_with_prob
        assert m_with_prob["roc_auc"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2×2 for binary classification."""
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_partial_correct(self):
        """Sanity check on a partly-correct set."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # 3/4 correct
        m = compute_metrics(y_true, y_pred)
        assert 0.0 < m["accuracy"] < 1.0
