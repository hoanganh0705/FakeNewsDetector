import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


class TestComputeMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_metrics_keys_present(self):
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
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])

        m_no_prob = compute_metrics(y_true, y_pred)
        m_with_prob = compute_metrics(y_true, y_pred, y_prob)

        assert "roc_auc" not in m_no_prob
        assert "roc_auc" in m_with_prob
        assert m_with_prob["roc_auc"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_partial_correct(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # 3/4 correct
        m = compute_metrics(y_true, y_pred)
        assert 0.0 < m["accuracy"] < 1.0
