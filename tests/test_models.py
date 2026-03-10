"""Tests for model architecture definitions (BiLSTM & PhoBERT).

PhoBERT tests are skipped when transformers model download would be required;
BiLSTM tests run purely on small tensors with no I/O.
"""

import numpy as np
import pytest
import torch

from src.models.bilstm_model import BiLSTMClassifier


# ──────────────────────────────────────────────────────────────
# BiLSTM
# ──────────────────────────────────────────────────────────────

class TestBiLSTMClassifier:
    """Forward-pass shape and basic behaviour checks."""

    VOCAB, EMB, HID, CLASSES = 500, 64, 32, 2

    def _make_model(self, **kw):
        defaults = dict(
            vocab_size=self.VOCAB,
            embedding_dim=self.EMB,
            hidden_dim=self.HID,
            num_classes=self.CLASSES,
            num_layers=1,
            dropout=0.0,
        )
        defaults.update(kw)
        return BiLSTMClassifier(**defaults)

    def test_output_shape(self):
        model = self._make_model()
        x = torch.randint(0, self.VOCAB, (4, 20))
        out = model(x)
        assert out.shape == (4, self.CLASSES)

    def test_output_shape_with_mask(self):
        model = self._make_model()
        x = torch.randint(0, self.VOCAB, (4, 20))
        mask = torch.ones(4, 20, dtype=torch.long)
        mask[:, 15:] = 0  # last 5 tokens are padding
        out = model(x, attention_mask=mask)
        assert out.shape == (4, self.CLASSES)

    def test_single_sample(self):
        model = self._make_model()
        x = torch.randint(0, self.VOCAB, (1, 10))
        out = model(x)
        assert out.shape == (1, self.CLASSES)

    def test_unidirectional(self):
        model = self._make_model(bidirectional=False)
        x = torch.randint(0, self.VOCAB, (2, 15))
        out = model(x)
        assert out.shape == (2, self.CLASSES)

    def test_load_pretrained_embeddings_numpy(self):
        model = self._make_model()
        matrix = np.random.randn(self.VOCAB, self.EMB).astype(np.float32)
        model.load_pretrained_embeddings(matrix)
        np.testing.assert_allclose(
            model.embedding.weight.data.numpy()[:self.VOCAB, :self.EMB],
            matrix,
            atol=1e-6,
        )

    def test_load_pretrained_embeddings_torch(self):
        model = self._make_model()
        matrix = torch.randn(self.VOCAB, self.EMB)
        model.load_pretrained_embeddings(matrix)
        assert torch.allclose(model.embedding.weight.data, matrix, atol=1e-6)

    def test_gradient_flows(self):
        """Backward pass should produce non-None gradients in embedding."""
        model = self._make_model()
        x = torch.randint(0, self.VOCAB, (2, 10))
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.embedding.weight.grad is not None


# ──────────────────────────────────────────────────────────────
# PhoBERT — heavier; skip if model weights are unavailable
# ──────────────────────────────────────────────────────────────

class TestPhoBertClassifier:
    """Forward-pass shape checks for PhoBertClassifier.

    Skipped entirely if the PhoBERT weights cannot be loaded (e.g. no
    internet / no local cache), so CI without GPU/model files still passes.
    """

    @pytest.fixture(autouse=True)
    def _load_model(self):
        try:
            from src.models.phobert_model import PhoBertClassifier
            self.model = PhoBertClassifier(num_classes=2, dropout=0.0)
        except Exception:
            pytest.skip("PhoBERT model weights unavailable — skipping")

    def test_output_shape(self):
        ids = torch.randint(0, 1000, (2, 16))
        mask = torch.ones_like(ids)
        out = self.model(ids, mask)
        assert out.shape == (2, 2)

    def test_single_sample(self):
        ids = torch.randint(0, 1000, (1, 8))
        mask = torch.ones_like(ids)
        out = self.model(ids, mask)
        assert out.shape == (1, 2)
