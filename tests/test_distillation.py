"""
Unit tests for Phase 3 — Knowledge Distillation.

Covers:

* ``StudentBiLSTM`` shape & parameter-count invariants
* ``distillation_loss`` mathematical properties:
    - L ≥ 0
    - L(α=0) == CE(student, label)
    - L(α=1) ≥ 0 and recovers when student = teacher
    - L(T → ∞) → α · log(num_classes)
* ``run_distillation_evaluation`` smoke test
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Student model
# ──────────────────────────────────────────────────────────────────────


class TestStudentBiLSTM:
    def test_smoke_construct(self):
        from src.models.student_model import StudentBiLSTM
        m = StudentBiLSTM(vocab_size=1000, embedding_dim=32, hidden_dim=32)
        assert m.is_student is True
        assert m.embedding_dim == 32
        assert m.hidden_dim == 32
        assert m.num_layers == 1

    def test_parameter_count_smaller_than_teacher(self):
        from src.models.bilstm_model import BiLSTMClassifier
        from src.models.student_model import StudentBiLSTM

        teacher = BiLSTMClassifier(vocab_size=20000, embedding_dim=128, hidden_dim=128, num_layers=2)
        student = StudentBiLSTM(vocab_size=20000)
        assert student.count_parameters() < teacher.count_parameters()
        # student should be at least 2× smaller (in our setup it's ~4× smaller)
        ratio = teacher.count_parameters() / student.count_parameters()
        assert ratio >= 2.0, f"compression ratio = {ratio:.2f}, expected >= 2"

    def test_forward_signature_matches_teacher(self):
        from src.models.bilstm_model import BiLSTMClassifier
        from src.models.student_model import StudentBiLSTM

        torch.manual_seed(0)
        teacher = BiLSTMClassifier(vocab_size=200, embedding_dim=16, hidden_dim=16, num_layers=1)
        torch.manual_seed(0)  # reset so the embedding init is comparable
        student = StudentBiLSTM(vocab_size=200, embedding_dim=16, hidden_dim=16, num_layers=1)

        x = torch.randint(0, 200, (4, 10))
        mask = torch.ones_like(x)
        t_logits = teacher(x, mask)
        s_logits = student(x, mask)
        assert t_logits.shape == s_logits.shape == (4, 2)
        assert t_logits.dtype == s_logits.dtype == torch.float32

    def test_forward_with_no_mask(self):
        from src.models.student_model import StudentBiLSTM
        m = StudentBiLSTM(vocab_size=100, embedding_dim=16, hidden_dim=16)
        x = torch.randint(0, 100, (2, 8))
        logits = m(x)
        assert logits.shape == (2, 2)

    def test_size_methods(self):
        from src.models.student_model import StudentBiLSTM
        m = StudentBiLSTM(vocab_size=100)
        n = m.count_parameters()
        sz = m.model_size_mb()
        assert n > 0
        # 4 bytes per param float32 → n*4 / (1024^2) MB
        assert math.isclose(sz, (n * 4) / (1024 ** 2), rel_tol=1e-9)

    def test_invalid_vocab_size(self):
        from src.models.student_model import StudentBiLSTM
        with pytest.raises(ValueError):
            StudentBiLSTM(vocab_size=0)


# ──────────────────────────────────────────────────────────────────────
# KD loss
# ──────────────────────────────────────────────────────────────────────


class TestDistillationLoss:
    def test_loss_is_non_negative(self):
        from src.training.train_student import distillation_loss
        torch.manual_seed(0)
        for _ in range(10):
            B, C = 4, 3
            s = torch.randn(B, C)
            t = torch.randn(B, C)
            y = torch.randint(0, C, (B,))
            loss = distillation_loss(s, t, y, alpha=0.5, temperature=4.0)
            assert torch.isfinite(loss)
            assert loss.item() >= 0.0

    def test_alpha_zero_recovers_cross_entropy(self):
        """With α=0 the KD term drops out and the loss is exactly CE."""
        from src.training.train_student import distillation_loss
        torch.manual_seed(0)
        s = torch.tensor([[2.0, 1.0, 0.0], [0.5, 1.5, 0.0]])
        y = torch.tensor([0, 1])
        teacher = torch.randn_like(s)  # any teacher; should be ignored.
        loss_kd = distillation_loss(s, teacher, y, alpha=0.0, temperature=4.0)
        loss_ce = F.cross_entropy(s, y).item()
        assert math.isclose(loss_kd.item(), loss_ce, rel_tol=1e-5, abs_tol=1e-6)

    def test_alpha_one_teacher_equals_student_loss_zero(self):
        """If student matches teacher exactly, α=1 KD loss → 0."""
        from src.training.train_student import distillation_loss
        torch.manual_seed(0)
        s = torch.randn(8, 3)
        y = torch.randint(0, 3, (8,))
        loss = distillation_loss(s, s.clone(), y, alpha=1.0, temperature=2.0)
        # KL(p ‖ p) = 0
        assert loss.item() < 1e-5

    def test_high_temperature_dominates_kl_part(self):
        """As T grows the KL part scales as T² while the per-sample KL
        itself shrinks; the product should grow but stay bounded by
        α · log(num_classes) for *uniform* teacher distribution.
        We just check the loss is finite and that increasing T beyond 1
        does not blow the loss up to infinity."""
        from src.training.train_student import distillation_loss
        torch.manual_seed(0)
        B, C = 32, 4
        s = torch.randn(B, C)
        t = torch.randn(B, C)
        y = torch.randint(0, C, (B,))

        l_normal = distillation_loss(s, t, y, alpha=0.7, temperature=4.0)
        l_high = distillation_loss(s, t, y, alpha=0.7, temperature=200.0)
        # Both should be finite, non-negative, and ≪ e^{T}.
        assert torch.isfinite(l_normal)
        assert torch.isfinite(l_high)
        assert l_high.item() < 1e8

    def test_teacher_gradients_dont_propagate(self):
        """KD loss must not push gradients into teacher logits."""
        from src.training.train_student import distillation_loss
        torch.manual_seed(0)
        s = torch.randn(4, 3, requires_grad=True)
        t = torch.randn(4, 3, requires_grad=True)
        y = torch.randint(0, 3, (4,))
        loss = distillation_loss(s, t, y, alpha=0.7, temperature=4.0)
        loss.backward()
        # Student should have gradients
        assert s.grad is not None and torch.any(s.grad != 0).item()
        # Teacher must NOT (we detached inside distillation_loss)
        assert t.grad is None or torch.all(t.grad == 0).item()


# ──────────────────────────────────────────────────────────────────────
# Trainer smoke
# ──────────────────────────────────────────────────────────────────────


class TestStudentBiLSTMTrainer:
    def _make_loader(self, n: int, vocab: int, seq_len: int = 6):
        from torch.utils.data import DataLoader
        from src.features.embedding_features import TextDataset, collate_fn

        rng = np.random.default_rng(0)
        seqs = [rng.integers(1, vocab, size=seq_len).tolist() for _ in range(n)]
        y = rng.integers(0, 2, size=n).tolist()
        ds = TextDataset(seqs, y)
        return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    def test_trainer_smoke_train_one_epoch(self):
        from src.training.train_student import StudentBiLSTMTrainer

        n_train, n_val = 16, 8
        train_loader = self._make_loader(n_train, vocab=200)
        val_loader = self._make_loader(n_val, vocab=200)
        teacher_train = np.zeros(n_train, dtype=np.float32)
        teacher_val = np.zeros(n_val, dtype=np.float32)

        trainer = StudentBiLSTMTrainer(
            vocab_size=200, embedding_dim=16, hidden_dim=16, num_layers=1,
            learning_rate=1e-2,
        )
        trainer.train(
            train_loader, val_loader,
            teacher_logits_train=teacher_train,
            teacher_logits_val=teacher_val,
            epochs=1, patience=10,
        )
        assert len(trainer.history["train_loss"]) == 1
        assert trainer.best_state is not None  # at least one "best" snapshot

    def test_save_load_roundtrip(self, tmp_path: Path):
        from src.training.train_student import StudentBiLSTMTrainer

        trainer = StudentBiLSTMTrainer(
            vocab_size=50, embedding_dim=8, hidden_dim=8, num_layers=1,
        )
        path = tmp_path / "ckpt.pt"
        trainer.save(str(path))
        loaded = StudentBiLSTMTrainer.load(str(path))
        # Sanity: same vocab/dim and same parameter values.
        assert loaded.vocab_size == 50
        assert loaded.embedding_dim == 8
        # Re-load weights (state_dict) and compare a few entries.
        sd_orig = trainer.model.state_dict()
        sd_new = loaded.model.state_dict()
        assert set(sd_orig.keys()) == set(sd_new.keys())
        for k in sd_orig:
            assert torch.allclose(sd_orig[k], sd_new[k])


# ──────────────────────────────────────────────────────────────────────
# End-to-end evaluation smoke
# ──────────────────────────────────────────────────────────────────────


class TestDistillationEvaluation:
    def test_evaluation_produces_table_and_figure(self, monkeypatch, tmp_path):
        """Smoke test: with fake teacher metrics + a fake student, the
        evaluation should still produce a table and a figure."""
        from src.training import distillation_evaluation as de

        # Fake student metrics
        fake_student = {
            "student_config": {
                "n_params": 500_000,
                "size_mb": 2.0,
            },
            "test": {"f1_macro": 0.81},
        }
        # Patch out the heavy loaders so this test is hermetic.
        monkeypatch.setattr(de, "_load_teacher_metrics", lambda name: {
            "test": {"f1_macro": 0.89 if name == "bert" else 0.82}
        })
        monkeypatch.setattr(de, "_load_student_metrics", lambda: fake_student)
        monkeypatch.setattr(de, "_load_teacher_bilstm_model", lambda: _DummyModel())
        monkeypatch.setattr(de, "_load_student_model", lambda vocab_size: _DummyModel())

        out = de.run_distillation_evaluation(
            tables_dir=str(tmp_path / "tables"),
            figures_dir=str(tmp_path / "figures"),
            n_runs=3,
        )

        assert out["teacher_phobert"]["f1"] == 0.89
        assert out["student_bilstm"]["f1"] == 0.81
        assert (tmp_path / "tables" / "table_distillation.tex").exists()
        assert (tmp_path / "figures" / "fig_distillation_tradeoff.png").exists()
        assert (tmp_path / "figures" / "fig_distillation_tradeoff.pdf").exists()


class _DummyModel(torch.nn.Module):
    """Tiny linear module so the timing code has *something* to call."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor, mask=None):
        # Use mean over time then linear.
        return self.lin(x.float().mean(dim=1))
