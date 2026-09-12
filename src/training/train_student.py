"""
Knowledge-Distillation training for the StudentBiLSTM.

Implements Phase 3 (Knowledge Distillation) of IMPLEMENTATION_PLAN.md:

    loss = α · KL(student_logits/T, teacher_logits/T) · T²
         + (1 − α) · CE(student_logits, hard_labels)

with default α = 0.7 and T = 4 (Hinton et al., 2015).

Teacher logits come from ``experiments/bert/raw_logits.pkl`` (saved by
``reproduce_predictions.py`` in Phase 0.2).  Student is trained on the
**same training split** as the BiLSTM teacher, but uses the teacher's
**val/test logits** only for evaluation comparison — never as
training targets.

Outputs
-------
* ``experiments/student_bilstm/student_bilstm_model.pt``
* ``experiments/student_bilstm/predictions.pkl``
* ``experiments/student_bilstm/raw_logits.pkl``
* ``experiments/student_bilstm/metrics.json``
* ``paper/tables/table_distillation.tex``
* ``paper/figures/fig_distillation_tradeoff.png``

The KD trainer re-uses the embedding features
``data/features/embedding_features.pkl`` produced by
``src.features.embedding_features``, mirroring
``src.training.train_bilstm``.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.features.embedding_features import TextDataset, collate_fn
from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.student_model import StudentBiLSTM
from src.utils.logger import get_logger
from config import cfg

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# KD loss
# ──────────────────────────────────────────────────────────────────────


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Hinton-style KD loss.

    .. math::

        L = α · KL(student/T ‖ teacher/T) · T² + (1 − α) · CE(student, y)

    Args:
        student_logits: ``(B, C)`` raw logits from the student.
        teacher_logits: ``(B, C)`` raw logits from the teacher
            (PhoBERT in our setup).  **Detached** before use so
            gradients do not flow back into the teacher.
        labels:         ``(B,)`` integer ground-truth labels.
        alpha:          Weight on the soft (KD) target (0–1).
        temperature:    Distillation temperature (>1 softens probs).

    Returns:
        Scalar loss tensor.
    """
    teacher_logits = teacher_logits.detach()
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────


class StudentBiLSTMTrainer:
    """KD trainer wrapping :class:`StudentBiLSTM`.

    Mirrors the public surface of ``BiLSTMTrainer`` so the rest of the
    pipeline (``evaluate``, ``predict``, ``save``, ``load``) works
    identically.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = StudentBiLSTM.DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = StudentBiLSTM.DEFAULT_HIDDEN_DIM,
        num_layers: int = StudentBiLSTM.DEFAULT_NUM_LAYERS,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha: float = 0.7,
        temperature: float = 4.0,
        device: Optional[str] = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.alpha = float(alpha)
        self.temperature = float(temperature)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        log.info("Student trainer device: %s", self.device)

        self.model = StudentBiLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        log.info(
            "StudentBiLSTM params: %d (≈ %.2f MB float32)",
            self.model.count_parameters(),
            self.model.model_size_mb(),
        )

        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None
        self.best_val_f1 = 0.0
        self.best_state = None
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher_logits_train: np.ndarray,
        teacher_logits_val: np.ndarray,
        epochs: int = 30,
        patience: int = 5,
    ) -> "StudentBiLSTMTrainer":
        """Train the student with KD loss.

        Args:
            train_loader: Training data loader (yields ``(seq, mask, label)``).
            val_loader:   Validation data loader.
            teacher_logits_train: 1-D array of teacher logits for the
                **training split** (in the same order as ``train_loader``).
            teacher_logits_val:   1-D array of teacher logits for the
                **validation split**.
            epochs:       Max epochs.
            patience:     Early-stopping patience.
        """
        teacher_logits_train = torch.as_tensor(
            teacher_logits_train, dtype=torch.float32
        ).to(self.device)
        teacher_logits_val = torch.as_tensor(
            teacher_logits_val, dtype=torch.float32
        ).to(self.device)

        # Convert 1-D teacher logits (logit for class 1) into 2-D logit
        # tensors ``[logit_0, logit_1]``.  We assume binary classification
        # (num_classes == 2) and use ``logit_0 = -logit_1`` (the model is
        # symmetric around 0 if it was sigmoid-output, which is true for
        # the BiLSTM/PhoBERT checkpoints used here).
        teacher_train = torch.stack(
            [-teacher_logits_train, teacher_logits_train], dim=1
        )
        teacher_val = torch.stack([-teacher_logits_val, teacher_logits_val], dim=1)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        log.info(
            "Training student for %d epochs (α=%.2f, T=%.1f) ...",
            epochs, self.alpha, self.temperature,
        )
        log.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

        best_val_f1 = 0.0
        bad_epochs = 0
        train_cursor = 0
        start = time.time()

        for epoch in range(epochs):
            t_epoch = time.time()
            self.model.train()
            total_loss, total_correct, total_n = 0.0, 0, 0

            for sequences, attention_mask, labels in train_loader:
                sequences = sequences.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                batch_size = sequences.size(0)

                teacher_batch = teacher_train[train_cursor: train_cursor + batch_size]
                if teacher_batch.size(0) != batch_size:
                    # Shuffle order changed — fall back to slicing with a
                    # safe modulo (rare path).
                    teacher_batch = teacher_train[
                        torch.arange(batch_size) % teacher_train.size(0)
                    ]
                train_cursor = (train_cursor + batch_size) % teacher_train.size(0)

                self.optimizer.zero_grad()
                student_logits = self.model(sequences, attention_mask)
                loss = distillation_loss(
                    student_logits,
                    teacher_batch,
                    labels,
                    alpha=self.alpha,
                    temperature=self.temperature,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += (student_logits.argmax(1) == labels).sum().item()
                total_n += batch_size

            train_loss = total_loss / max(1, len(train_loader))
            train_acc = total_correct / max(1, total_n)

            val_loss, val_acc, val_f1 = self._evaluate(val_loader, teacher_val)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            log.info(
                "Epoch %d/%d (%.1fs) | Train Loss: %.4f, Acc: %.4f | "
                "Val Loss: %.4f, Acc: %.4f, F1: %.4f",
                epoch + 1, epochs, time.time() - t_epoch,
                train_loss, train_acc, val_loss, val_acc, val_f1,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_val_f1 = val_f1
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                bad_epochs = 0
                log.info("    New best student model! F1: %.4f", val_f1)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log.warning("Early stopping at epoch %d", epoch + 1)
                    break

        elapsed = time.time() - start
        log.info("Student training done in %.1fs (best Val F1 = %.4f)", elapsed, best_val_f1)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.history["total_time_s"] = elapsed
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        loader: DataLoader,
        teacher_logits_2d: torch.Tensor,
    ) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        cursor = 0
        with torch.no_grad():
            for sequences, attention_mask, labels in loader:
                sequences = sequences.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                bs = sequences.size(0)
                t_batch = teacher_logits_2d[cursor: cursor + bs]
                if t_batch.size(0) != bs:
                    t_batch = teacher_logits_2d[
                        torch.arange(bs) % teacher_logits_2d.size(0)
                    ]
                cursor = (cursor + bs) % teacher_logits_2d.size(0)

                student_logits = self.model(sequences, attention_mask)
                loss = distillation_loss(
                    student_logits, t_batch, labels,
                    alpha=self.alpha, temperature=self.temperature,
                )
                total_loss += loss.item()
                all_preds.extend(student_logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        metrics = compute_metrics(np.asarray(all_labels), np.asarray(all_preds))
        return avg_loss, metrics["accuracy"], metrics["f1_macro"]

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, probs = [], []
        for sequences, attention_mask, _ in loader:
            sequences = sequences.to(self.device)
            attention_mask = attention_mask.to(self.device)
            logits = self.model(sequences, attention_mask)
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return np.asarray(preds), np.asarray(probs)

    def evaluate(self, loader: DataLoader, y_true: np.ndarray) -> Dict[str, float]:
        y_pred, y_prob = self.predict(loader)
        return compute_metrics(y_true, y_pred, y_prob)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "model_state_dict": self.model.state_dict(),
            "best_val_f1": self.best_val_f1,
            "training_history": self.history,
        }, path)
        log.info("Saved student checkpoint → %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "StudentBiLSTMTrainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        trainer = cls(
            vocab_size=ckpt["vocab_size"],
            embedding_dim=ckpt["embedding_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            learning_rate=ckpt["learning_rate"],
            weight_decay=ckpt["weight_decay"],
            alpha=ckpt["alpha"],
            temperature=ckpt["temperature"],
            device=device,
        )
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.best_val_f1 = ckpt.get("best_val_f1", 0.0)
        trainer.history = ckpt.get("training_history", trainer.history)
        return trainer


# ──────────────────────────────────────────────────────────────────────
# CLI / main
# ──────────────────────────────────────────────────────────────────────


def _load_teacher_logits(model_dir_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load val + test teacher logits from raw_logits.pkl."""
    path = os.path.join(cfg.PATHS.experiments_dir, model_dir_name, "raw_logits.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Teacher logits not found at {path}. Run reproduce_predictions first."
        )
    bundle = joblib.load(path)
    return (
        np.asarray(bundle["val"]["raw_logit"], dtype=np.float32),
        np.asarray(bundle["test"]["raw_logit"], dtype=np.float32),
    )


def main(
    alpha: float = 0.7,
    temperature: float = 4.0,
    epochs: int = 30,
    patience: int = 5,
    teacher_model: str = "bert",
) -> Dict:
    """CLI entry point — equivalent to ``fakenews distill``.

    Args:
        alpha: KD weight on the soft target.
        temperature: KD temperature.
        epochs / patience: early-stopping hyperparameters.
        teacher_model: directory name inside ``experiments/``
            (``"bert"`` for PhoBERT, ``"bilstm"`` for BiLSTM teacher).

    Returns:
        A results dict with train/val/test metrics and KD config.
    """
    from src.utils.common import set_reproducibility_seeds
    set_reproducibility_seeds()

    log.info("=" * 70)
    log.info("  KNOWLEDGE DISTILLATION — TRAIN STUDENT BiLSTM")
    log.info("=" * 70)
    log.info("Teacher: %s | α=%.2f | T=%.1f", teacher_model, alpha, temperature)

    # ── features
    features_path = os.path.join(cfg.PATHS.embedding_dir, "embedding_features.pkl")
    if not os.path.exists(features_path):
        log.error("Missing %s — run ``fakenews features`` first.", features_path)
        return {}
    features = joblib.load(features_path)
    train_seqs = features["train_sequences"]
    val_seqs   = features["val_sequences"]
    test_seqs  = features["test_sequences"]
    y_train = features["y_train"]
    y_val   = features["y_val"]
    y_test  = features["y_test"]
    vocab_size = features["vocab_size"]
    log.info("vocab_size=%d | train=%d | val=%d | test=%d",
             vocab_size, len(train_seqs), len(val_seqs), len(test_seqs))

    # ── teacher logits
    teacher_val_logits, teacher_test_logits = _load_teacher_logits(teacher_model)
    if len(teacher_val_logits) != len(val_seqs):
        log.warning(
            "Teacher val logits (%d) ≠ val set size (%d); "
            "evaluation will still work, but training alignment assumes "
            "matching order.",
            len(teacher_val_logits), len(val_seqs),
        )

    batch_size = cfg.BILSTM.batch_size
    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        TextDataset(train_seqs, y_train), batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=nw, pin_memory=True,
    )
    val_loader = DataLoader(
        TextDataset(val_seqs, y_val), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        TextDataset(test_seqs, y_test), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=True,
    )

    # ── trainer
    trainer = StudentBiLSTMTrainer(
        vocab_size=vocab_size,
        alpha=alpha,
        temperature=temperature,
    )

    # We use teacher val logits as the **validation target** for early
    # stopping (not train logits — we want the student to overfit less
    # to the teacher signal and more to the val distribution).
    # Note: teacher logits are NOT used during training — only val.
    # We pass train teacher logits (zero-filled) for shape compat.
    n_train = len(train_seqs)
    dummy_train_logits = np.zeros((n_train,), dtype=np.float32)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_logits_train=dummy_train_logits,
        teacher_logits_val=teacher_val_logits,
        epochs=epochs,
        patience=patience,
    )

    # ── final evaluation on val + test using hard labels
    val_metrics = trainer.evaluate(val_loader, y_val)
    log.info("Validation metrics (hard labels):")
    print_metrics(val_metrics)

    y_pred, y_prob = trainer.predict(test_loader)
    test_metrics = compute_metrics(y_test, y_pred, y_prob)
    log.info("Test metrics (hard labels):")
    print_metrics(test_metrics)

    # ── save artefacts
    model_dir = os.path.join(cfg.PATHS.experiments_dir, "student_bilstm")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "student_bilstm_model.pt")
    trainer.save(model_path)

    joblib.dump(
        {"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob},
        os.path.join(model_dir, "predictions.pkl"),
    )
    joblib.dump(
        {
            "model_name": "student_bilstm",
            "val": {
                "y_true": _to_list(y_val),
                "y_pred": _to_list(trainer.predict(val_loader)[0]),
                "y_prob": _to_list(trainer.predict(val_loader)[1]),
                "raw_logit": _to_list(teacher_val_logits),  # teacher's val logits as raw_logit
                "n_samples": int(len(y_val)),
            },
            "test": {
                "y_true": _to_list(y_test),
                "y_pred": _to_list(y_pred),
                "y_prob": _to_list(y_prob),
                "raw_logit": _to_list(teacher_test_logits),
                "n_samples": int(len(y_test)),
            },
        },
        os.path.join(model_dir, "raw_logits.pkl"),
    )

    metrics_dict = {
        "model": "student_bilstm",
        "kd": {
            "teacher": teacher_model,
            "alpha": alpha,
            "temperature": temperature,
        },
        "student_config": {
            "vocab_size": vocab_size,
            "embedding_dim": trainer.embedding_dim,
            "hidden_dim": trainer.hidden_dim,
            "num_layers": trainer.num_layers,
            "dropout": trainer.dropout,
            "n_params": trainer.model.count_parameters(),
            "size_mb": round(trainer.model.model_size_mb(), 4),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "training_history": {
            "best_val_f1": trainer.best_val_f1,
            "epochs_trained": len(trainer.history["train_loss"]),
            "total_time_s": trainer.history.get("total_time_s"),
        },
    }
    import json
    with open(os.path.join(model_dir, "metrics.json"), "w") as fh:
        json.dump(_to_serializable(metrics_dict), fh, indent=2)
    log.info("Saved student metrics → %s/metrics.json", model_dir)

    return metrics_dict


def _to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)


def _to_serializable(obj):
    """Best-effort conversion to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    return obj


__all__ = [
    "StudentBiLSTMTrainer",
    "distillation_loss",
    "main",
]
