from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.analysis.method_agreement import rank_agreement
from src.analysis.bilstm_attribution import (
    bilstm_simple_gradients,
    bilstm_ig,
)
from src.analysis.method_agreement import faithfulness
from src.models.bilstm_model import BiLSTMClassifier

class _LinearMockModel:
    def __init__(self, weights: np.ndarray):
        self.coef_ = np.stack([-weights, weights], axis=0)  # (2, D)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef_[1]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(x, "toarray"):
            x = x.toarray()
        logits = x @ self.coef_[1]                  # (N,)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - probs, probs], axis=1)


class _ToyVectorizer:
    def __init__(self, vocab: list[str]):
        self._vocab = vocab

    def transform(self, texts):
        rows = []
        for text in texts:
            tokens = str(text).split()
            row = np.array([1.0 if w in tokens else 0.0 for w in self._vocab],
                           dtype=np.float32)
            rows.append(row)
        return _Sparse(rows)

    def get_feature_names_out(self):
        return np.array(self._vocab)


class _Sparse:
    def __init__(self, rows):
        self._rows = rows

    def todense(self):
        return np.asarray(self._rows)


def _toy_bilstm(vocab_size: int = 20, emb_dim: int = 16, hidden: int = 16):
    torch.manual_seed(0)
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        hidden_dim=hidden,
        num_classes=2,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    return model


class _ToyVocab:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, words):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}


class TestShapEfficiency:
    def test_synthetic_kernel_shap_matches_linear_truth(self):
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        from src.analysis.lr_svm_shap import lr_kernel_shap

        vocab = ["alpha", "beta", "gamma"]
        weights = np.array([1.0, -2.0, 0.0])
        model = _LinearMockModel(weights)
        vec = _ToyVectorizer(vocab)
        text = "alpha beta"

        toks, sv = lr_kernel_shap(text, model, vec, n_samples=200)

        alpha_idx = toks.index("alpha")
        beta_idx = toks.index("beta")
        assert math.isclose(sv[alpha_idx], 1.0, abs_tol=0.4)
        assert math.isclose(sv[beta_idx], -2.0, abs_tol=0.4)

    def test_svm_linear_shap_exact_for_linear_svc(self):
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        from src.analysis.lr_svm_shap import svm_linear_shap

        vocab = ["foo", "bar", "baz"]
        model = _LinearMockModel(np.array([0.7, -1.3, 0.0]))

        toks, sv = svm_linear_shap("foo bar", model, _ToyVectorizer(vocab))
        assert "foo" in toks and "bar" in toks
        foo_idx = toks.index("foo")
        bar_idx = toks.index("bar")
        assert math.isclose(sv[foo_idx], 0.7, abs_tol=1e-4)
        assert math.isclose(sv[bar_idx], -1.3, abs_tol=1e-4)


class TestIntegratedGradients:
    def test_ig_zeros_for_zero_embedding_contribution(self):
        try:
            import captum  # noqa: F401
        except ImportError:
            pytest.skip("captum not installed")

        vocab = _ToyVocab(["xin", "chào", "thế", "giới"])
        model = _toy_bilstm(vocab_size=len(vocab.word2idx), emb_dim=8, hidden=8)

        text = "xin chào thế giới"
        toks, ig_scores = bilstm_ig(text, model, vocab, max_len=8, n_steps=10)
        assert len(toks) == len(ig_scores)
        assert len(toks) == 4
        assert np.all(np.isfinite(ig_scores))
        assert np.all(ig_scores >= 0.0)

    def test_ig_reduces_to_vanilla_gradients_when_n_steps_small(self):
        try:
            import captum  # noqa: F401
        except ImportError:
            pytest.skip("captum not installed")

        vocab = _ToyVocab(["a", "b", "c"])
        model = _toy_bilstm(vocab_size=len(vocab.word2idx), emb_dim=8, hidden=8)
        text = "a b c"
        _, ig = bilstm_ig(text, model, vocab, max_len=4, n_steps=1)
        _, grad = bilstm_simple_gradients(text, model, vocab, max_len=4)
        ig_n = ig / (np.linalg.norm(ig) + 1e-12)
        grad_n = grad / (np.linalg.norm(grad) + 1e-12)
        cos = float(np.dot(ig_n, grad_n))
        assert cos > 0.0  # strictly positive


class TestRankAgreement:
    def test_identical_vectors_have_jaccard_one(self):
        scores = [0.4, -0.1, 0.9, 0.0, -0.5, 0.2]
        result = rank_agreement(scores, scores, top_k=3)
        assert result["jaccard"] == pytest.approx(1.0)
        assert result["intersection"] == 3
        assert result["union"] == 3

    def test_disjoint_vectors_have_jaccard_zero(self):
        a = [10.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0]
        result = rank_agreement(a, b, top_k=2)
        assert result["jaccard"] == 0.0

    def test_top_k_larger_than_length(self):
        a = [0.1, 0.2]
        b = [0.1, 0.2]
        result = rank_agreement(a, b, top_k=10)
        assert result["jaccard"] == pytest.approx(1.0)


class TestFaithfulness:
    def test_faithfulness_drops_when_important_token_removed(self):
        def predict_proba(text: str) -> np.ndarray:
            tokens = text.split()
            p_pos = 0.9 if "magic" in tokens else 0.1
            return np.array([[1.0 - p_pos, p_pos]])

        text = "this text contains magic which matters"
        words = text.split()
        attribution = [0.0 if w != "magic" else 10.0 for w in words]
        result = faithfulness(predict_proba, text, attribution, remove_top_k=1)
        assert result["p_orig"] == pytest.approx(0.9)
        assert result["p_masked"] == pytest.approx(0.1)
        assert result["drop"] == pytest.approx(0.8)
        assert result["masked_tokens"] == ["magic"]

    def test_faithfulness_zero_when_no_token_matters(self):
        def predict_proba(text: str) -> np.ndarray:
            return np.array([[0.5, 0.5]])

        text = "a b c d"
        attribution = [1.0, 1.0, 1.0, 1.0]
        result = faithfulness(predict_proba, text, attribution, remove_top_k=2)
        assert abs(result["drop"]) < 1e-9


class TestExplainabilityRunner:
    def test_pick_examples_returns_at_most_n(self):
        from src.analysis.explainability_runner import pick_examples

        examples = pick_examples(n_examples=5)
        assert isinstance(examples, list)
        assert len(examples) <= 5
        for ex in examples:
            assert {"id", "text", "true_label"}.issubset(ex.keys())


class TestVisualizationSmoke:
    def test_visualize_lr_svm_token_importance(self, tmp_path):
        from src.analysis.lr_svm_shap import visualize_token_importance

        out = tmp_path / "tokens.png"
        path = visualize_token_importance(
            tokens=["fake", "news", "report", "confirmed"],
            scores=[0.4, 0.3, -0.2, -0.5],
            save_path=str(out),
            top_k=3,
            title="LR/SVM",
        )
        assert (tmp_path / "tokens.png").exists()
        assert (tmp_path / "tokens.pdf").exists()
        assert path.endswith(".png")

    def test_visualize_bilstm_attribution(self, tmp_path):
        from src.analysis.bilstm_attribution import visualize_bilstm_attribution

        out = tmp_path / "bilstm.png"
        path = visualize_bilstm_attribution(
            text="đây là tin thật",
            attribution=[0.2, -0.1, 0.3, -0.4],
            save_path=str(out),
        )
        assert (tmp_path / "bilstm.png").exists()
        assert (tmp_path / "bilstm.pdf").exists()
        assert path.endswith(".png")

    def test_compare_attribution_methods(self, tmp_path):
        from src.analysis.phobert_attribution import compare_attribution_methods

        out = tmp_path / "compare.png"
        path = compare_attribution_methods(
            text="tin giả tràn ngập",
            methods_results=[
                ("SHAP",   (["tin", "giả", "tràn", "ngập"], np.array([0.4, 0.3, -0.1, -0.2]))),
                ("IG",     (["tin", "giả", "tràn", "ngập"], np.array([0.5, 0.4,  0.0, -0.3]))),
            ],
            save_path=str(out),
            title="PhoBERT comparison",
        )
        assert (tmp_path / "compare.png").exists()