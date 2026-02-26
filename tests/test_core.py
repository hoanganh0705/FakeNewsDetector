"""
Unit tests for core FakeNewsDetector modules.

Run with:
    pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# ─── ensure project root is on the path ───────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# =============================================================================
# tests/test_metrics.py
# =============================================================================

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
        y_pred = np.array([0, 1, 1, 1])   # 3/4 correct
        m = compute_metrics(y_true, y_pred)
        assert 0.0 < m["accuracy"] < 1.0


# =============================================================================
# tests/test_text_preprocessor.py
# =============================================================================

from src.preprocessing.text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor.clean_text()."""

    def setup_method(self):
        self.pp = TextPreprocessor()

    def test_removes_urls(self):
        text = "Đọc thêm tại https://example.com và www.foo.com hôm nay"
        result = self.pp.clean_text(text)
        assert "https://" not in result
        assert "www.foo.com" not in result

    def test_removes_html_tags(self):
        text = "<p>Tin tức <b>quan trọng</b></p>"
        result = self.pp.clean_text(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "tin tức" in result  # clean_text() lowercases input

    def test_removes_email(self):
        text = "Liên hệ qua email: contact@example.com để biết thêm."
        result = self.pp.clean_text(text)
        assert "contact@example.com" not in result

    def test_preserves_vietnamese_diacritics(self):
        text = "Chính phủ Việt Nam thông báo kế hoạch mới"
        result = self.pp.clean_text(text)
        assert "việt" in result  # lowercase preserved
        assert "chính" in result

    def test_empty_string_returns_empty(self):
        assert self.pp.clean_text("") == ""

    def test_non_string_returns_empty(self):
        assert self.pp.clean_text(None) == ""
        assert self.pp.clean_text(123) == ""

    def test_extra_whitespace_collapsed(self):
        text = "Tin   tức   hôm   nay"
        result = self.pp.clean_text(text)
        assert "  " not in result  # no double spaces

    def test_fit_transform_returns_matrix(self):
        texts = ["tin tức thật", "tin giả không đúng", "bài báo quan trọng"]
        X = self.pp.fit_transform(texts)
        assert X.shape[0] == 3   # 3 rows
        assert X.shape[1] > 0    # some features


# =============================================================================
# tests/test_statistical_tests.py
# =============================================================================

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


# =============================================================================
# tests/test_config.py
# =============================================================================

from config import cfg


class TestConfig:
    """Smoke-tests for the centralised config object."""

    def test_random_state_is_42(self):
        assert cfg.RANDOM_STATE == 42

    def test_split_ratios_sum_to_one(self):
        total = cfg.DATA.train_ratio + cfg.DATA.val_ratio + cfg.DATA.test_ratio
        assert total == pytest.approx(1.0)

    def test_paths_are_strings(self):
        assert isinstance(cfg.PATHS.raw_data, str)
        assert isinstance(cfg.PATHS.splits_dir, str)

    def test_bilstm_embedding_dim(self):
        assert cfg.BILSTM.embedding_dim == 256

    def test_phobert_model_name(self):
        assert cfg.PHOBERT.model_name == "vinai/phobert-base"
