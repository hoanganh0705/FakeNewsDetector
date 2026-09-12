"""
Phase 0 — reproduce_predictions.py

End-to-end reproduction of the 4 models (LR, SVM, BiLSTM, PhoBERT) used
in the Vietnamese Fake News Detector paper, with **on-disk artefacts
that match the format needed by Phase 1/2/3 downstream analyses**.

Stages
------
Stage A — sklearn models (LR + SVM)
    Re-loads the existing ``experiments/{lr,svm}/*.pkl`` checkpoints
    and re-runs inference on val + test to capture ``raw_logits.pkl``.
    No retraining, no network access.

Stage B — BiLSTM (train from scratch)
    Loads ``data/features/embedding_features.pkl`` +
    ``embedding_extractor.pkl``, builds ``BiLSTMClassifier`` per
    ``cfg.BILSTM``, trains with early stopping, saves model + predictions
    + raw_logits to ``experiments/bilstm/``.

Stage C — PhoBERT (fine-tune)
    Loads ``data/features/phobert_features.pkl``, builds ``PhoBertClassifier``,
    fine-tunes ``vinai/phobert-base`` for 3 epochs (or until early stop),
    saves model + predictions + raw_logits to ``experiments/bert/``.

Outputs (per model, in ``experiments/<model>/``)
------------------------------------------------
- ``predictions.pkl``     — ``{y_true, y_pred, y_prob}``  (test split)
- ``raw_logits.pkl``      — ``{val: {...}, test: {...}}``  with raw_logit per sample
- ``metrics.json``        — val + test metrics dict (overwrites prior)
- ``<model>_model.pt|pkl``— checkpoint (BiLSTM/PhoBERT only)
- ``experiment_log.json`` — append a Phase-0 entry

Usage
-----
    # Full retrain (sklearn + BiLSTM + PhoBERT), idempotent:
    python src/training/reproduce_predictions.py --stage all

    # Just one stage:
    python src/training/reproduce_predictions.py --stage A
    python src/training/reproduce_predictions.py --stage B
    python src/training/reproduce_predictions.py --stage C

    # Skip retraining of BiLSTM/PhoBERT (just re-load existing ckpts):
    python src/training/reproduce_predictions.py --stage all --skip-retrain

Notes
-----
- Random seeds are pinned via ``src.utils.common.set_reproducibility_seeds``.
- Deviations from paper numbers are expected (±1% F1) due to library
  version drift.  Documented in ``experiments/<model>/metrics.json``.
- For PhoBERT on CPU: expect 6-12 hours for 3 epochs.  GPU strongly
  recommended — use ``--device cuda`` if available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

# Make `config` importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from config import cfg  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.evaluation.metrics import compute_metrics, save_metrics  # noqa: E402

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Stage A — sklearn models (LR + SVM)
# ─────────────────────────────────────────────────────────────────────

def _to_list(arr):
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _resolve_raw_logit_sklearn(model, X: np.ndarray) -> np.ndarray:
    """Return pre-sigmoid / pre-platt score for any sklearn-style model."""
    # CalibratedClassifierCV path (linear SVM is wrapped)
    if hasattr(model, "calibrated_classifiers_"):
        base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        if base is None:
            raise RuntimeError("Cannot resolve base estimator of CalibratedClassifierCV.")
        if hasattr(base, "decision_function"):
            return np.asarray(base.decision_function(X), dtype=np.float64)
        raise RuntimeError(f"Base estimator {type(base).__name__} has no decision_function.")

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    raise RuntimeError(f"Cannot extract raw logits from {type(model).__name__}")


def stage_a(force: bool = False) -> None:
    log.info("=" * 70)
    log.info("Stage A — sklearn models (LR + SVM): capture raw_logits")
    log.info("=" * 70)

    for name, model_dir, model_path in [
        ("Logistic Regression", cfg.PATHS.lr_dir,  Path(cfg.PATHS.lr_dir)  / "lr_model.pkl"),
        ("SVM",                cfg.PATHS.svm_dir, Path(cfg.PATHS.svm_dir) / "svm_model.pkl"),
    ]:
        out_path = Path(model_dir) / "raw_logits.pkl"
        if out_path.exists() and not force:
            log.info("[%s] %s already exists — skipping (use --force to overwrite)",
                     name, out_path)
            continue

        log.info("[%s] loading features + model ...", name)
        feats = joblib.load(Path(cfg.PATHS.tfidf_dir) / "tfidf_features.pkl")
        bundle = joblib.load(model_path)
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

        combined = {"model_name": name, "timestamp": datetime.now().isoformat()}
        for split_name, X, y in [("val", feats["X_val"], feats["y_val"]),
                                 ("test", feats["X_test"], feats["y_test"])]:
            y_pred = np.asarray(model.predict(X))
            y_prob = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
            raw_logit = _resolve_raw_logit_sklearn(model, X)
            combined[split_name] = {
                "y_true": _to_list(y),
                "y_pred": _to_list(y_pred),
                "y_prob": _to_list(y_prob),
                "raw_logit": _to_list(raw_logit),
                "n_samples": int(len(y)),
            }
            metrics = compute_metrics(np.asarray(y), y_pred, y_prob)
            log.info("[%s/%s] acc=%.4f f1_macro=%.4f",
                     name, split_name, metrics["accuracy"], metrics["f1_macro"])

        joblib.dump(combined, out_path)
        log.info("[%s] saved → %s", name, out_path)


# ─────────────────────────────────────────────────────────────────────
# Stage B — BiLSTM (train from scratch)
# ─────────────────────────────────────────────────────────────────────

def stage_b(skip_retrain: bool = False, device: str = None, epochs: int = None) -> None:
    log.info("=" * 70)
    log.info("Stage B — BiLSTM (train from scratch)")
    log.info("=" * 70)

    from src.utils.common import set_reproducibility_seeds, compute_balanced_class_weights
    from torch.utils.data import DataLoader

    from src.features.embedding_features import TextDataset, collate_fn
    from src.models.bilstm_model import BiLSTMClassifier
    from src.training.train_bilstm import BiLSTMTrainer

    set_reproducibility_seeds()

    model_dir = Path(cfg.PATHS.bilstm_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt = model_dir / "bilstm_model.pt"
    pred_path = model_dir / "predictions.pkl"
    logits_path = model_dir / "raw_logits.pkl"

    feats = joblib.load(Path(cfg.PATHS.embedding_dir) / "embedding_features.pkl")
    train_seqs, val_seqs, test_seqs = feats["train_sequences"], feats["val_sequences"], feats["test_sequences"]
    y_train, y_val, y_test = feats["y_train"], feats["y_val"], feats["y_test"]
    vocab_size = int(feats["vocab_size"])

    bs = cfg.BILSTM.batch_size
    nw = min(4, os.cpu_count() or 1)

    def _loader(seqs, labels, shuffle):
        ds = TextDataset(seqs, labels)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn,
                          num_workers=nw, pin_memory=True)

    train_loader = _loader(train_seqs, y_train, True)
    val_loader = _loader(val_seqs, y_val, False)
    test_loader = _loader(test_seqs, y_test, False)

    if not skip_retrain or not model_ckpt.exists():
        log.info("Training BiLSTM (vocab=%d, bs=%d, epochs=%s) ...",
                 vocab_size, bs, epochs or cfg.BILSTM.epochs)
        trainer = BiLSTMTrainer(vocab_size=vocab_size, device=device)
        cw = None
        if cfg.BILSTM.class_weight == "balanced":
            cw = compute_balanced_class_weights(y_train)
        trainer.train(train_loader, val_loader, epochs=epochs, class_weights=cw)
        trainer.save(str(model_ckpt))
    else:
        log.info("Loading BiLSTM checkpoint from %s", model_ckpt)
        trainer = BiLSTMTrainer.load(str(model_ckpt), device=device)

    # ── Inference on val + test ──
    log.info("Running BiLSTM inference on val + test ...")
    val_pred, val_prob = trainer.predict(val_loader)
    val_logits = _extract_logits_bilstm(trainer, val_loader)  # shape (N, 2)
    test_pred, test_prob = trainer.predict(test_loader)
    test_logits = _extract_logits_bilstm(trainer, test_loader)

    # Save predictions.pkl (test split)
    joblib.dump({
        "y_true": _to_list(y_test),
        "y_pred": _to_list(test_pred),
        "y_prob": _to_list(test_prob),
    }, pred_path)

    # Save raw_logits.pkl (both splits)
    joblib.dump({
        "model_name": "BiLSTM",
        "timestamp": datetime.now().isoformat(),
        "val": {
            "y_true": _to_list(y_val),
            "y_pred": _to_list(val_pred),
            "y_prob": _to_list(val_prob),
            "raw_logit": _to_list(val_logits[:, 1]),  # logit for class 1
            "n_samples": int(len(y_val)),
        },
        "test": {
            "y_true": _to_list(y_test),
            "y_pred": _to_list(test_pred),
            "y_prob": _to_list(test_prob),
            "raw_logit": _to_list(test_logits[:, 1]),
            "n_samples": int(len(y_test)),
        },
    }, logits_path)

    # metrics.json
    val_metrics = compute_metrics(np.asarray(y_val), val_pred, val_prob)
    test_metrics = compute_metrics(np.asarray(y_test), test_pred, test_prob)
    save_metrics({
        "model": "BiLSTM",
        "phase": "Phase 0 reproduce",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vocab_size": vocab_size,
            "hidden_dim": cfg.BILSTM.hidden_dim,
            "num_layers": cfg.BILSTM.num_layers,
            "dropout": cfg.BILSTM.dropout,
            "batch_size": cfg.BILSTM.batch_size,
            "epochs": epochs or cfg.BILSTM.epochs,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "best_val_f1": float(trainer.best_val_f1),
    }, str(model_dir / "metrics.json"))

    log.info("BiLSTM test: acc=%.4f f1_macro=%.4f",
             test_metrics["accuracy"], test_metrics["f1_macro"])


def _extract_logits_bilstm(trainer, loader) -> np.ndarray:
    """Return raw logits (B, 2) for a BiLSTM trainer."""
    import torch.nn.functional as F
    trainer.model.eval()
    out = []
    with torch.no_grad():
        for sequences, attention_mask, _ in loader:
            sequences = sequences.to(trainer.device)
            attention_mask = attention_mask.to(trainer.device)
            logits = trainer.model(sequences, attention_mask)
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────
# Stage C — PhoBERT (fine-tune)
# ─────────────────────────────────────────────────────────────────────

def stage_c(skip_retrain: bool = False, device: str = None, epochs: int = None) -> None:
    log.info("=" * 70)
    log.info("Stage C — PhoBERT (fine-tune vinai/phobert-base)")
    log.info("=" * 70)

    from src.utils.common import set_reproducibility_seeds, compute_balanced_class_weights
    from torch.utils.data import DataLoader

    from src.features.phobert_features import PhoBertDataset
    from src.models.phobert_model import PhoBertClassifier
    from src.training.train_phobert import PhoBertTrainer

    set_reproducibility_seeds()

    model_dir = Path(cfg.PATHS.bert_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt = model_dir / "phobert_model.pt"
    pred_path = model_dir / "predictions.pkl"
    logits_path = model_dir / "raw_logits.pkl"

    feats = joblib.load(Path(cfg.PATHS.phobert_dir) / "phobert_features.pkl")
    y_train, y_val, y_test = feats["y_train"], feats["y_val"], feats["y_test"]

    bs = cfg.PHOBERT.batch_size
    nw = min(4, os.cpu_count() or 1)

    def _loader(ids, mask, labels, shuffle):
        ds = PhoBertDataset(ids, mask, labels)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=True)

    train_loader = _loader(feats["train_input_ids"], feats["train_attention_mask"], y_train, True)
    val_loader = _loader(feats["val_input_ids"], feats["val_attention_mask"], y_val, False)
    test_loader = _loader(feats["test_input_ids"], feats["test_attention_mask"], y_test, False)

    if not skip_retrain or not model_ckpt.exists():
        log.info("Fine-tuning PhoBERT (bs=%d, epochs=%s) ...", bs, epochs or cfg.PHOBERT.epochs)
        trainer = PhoBertTrainer(device=device)
        cw = compute_balanced_class_weights(y_train)
        trainer.train(train_loader, val_loader, epochs=epochs, class_weights=cw)
        trainer.save(str(model_ckpt))
    else:
        log.info("Loading PhoBERT checkpoint from %s", model_ckpt)
        trainer = PhoBertTrainer.load(str(model_ckpt), device=device)

    log.info("Running PhoBERT inference on val + test ...")
    val_pred, val_prob = trainer.predict(val_loader)
    val_logits = _extract_logits_phobert(trainer, val_loader)
    test_pred, test_prob = trainer.predict(test_loader)
    test_logits = _extract_logits_phobert(trainer, test_loader)

    joblib.dump({
        "y_true": _to_list(y_test),
        "y_pred": _to_list(test_pred),
        "y_prob": _to_list(test_prob),
    }, pred_path)

    joblib.dump({
        "model_name": "PhoBERT",
        "timestamp": datetime.now().isoformat(),
        "val": {
            "y_true": _to_list(y_val),
            "y_pred": _to_list(val_pred),
            "y_prob": _to_list(val_prob),
            "raw_logit": _to_list(val_logits[:, 1]),
            "n_samples": int(len(y_val)),
        },
        "test": {
            "y_true": _to_list(y_test),
            "y_pred": _to_list(test_pred),
            "y_prob": _to_list(test_prob),
            "raw_logit": _to_list(test_logits[:, 1]),
            "n_samples": int(len(y_test)),
        },
    }, logits_path)

    val_metrics = compute_metrics(np.asarray(y_val), val_pred, val_prob)
    test_metrics = compute_metrics(np.asarray(y_test), test_pred, test_prob)
    save_metrics({
        "model": "PhoBERT",
        "phase": "Phase 0 reproduce",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": cfg.PHOBERT.model_name,
            "max_seq_len": cfg.PHOBERT.max_seq_len,
            "batch_size": bs,
            "epochs": epochs or cfg.PHOBERT.epochs,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "best_val_f1": float(trainer.best_val_f1),
    }, str(model_dir / "metrics.json"))

    log.info("PhoBERT test: acc=%.4f f1_macro=%.4f",
             test_metrics["accuracy"], test_metrics["f1_macro"])


def _extract_logits_phobert(trainer, loader) -> np.ndarray:
    trainer.model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(trainer.device)
            mask = batch["attention_mask"].to(trainer.device)
            logits = trainer.model(ids, mask)
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 — reproduce all 4 models' predictions.pkl + raw_logits.pkl",
    )
    parser.add_argument("--stage", choices=["A", "B", "C", "all"], default="all",
                        help="Which stage to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing raw_logits.pkl even if present")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Re-load existing BiLSTM/PhoBERT checkpoints instead of retraining")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for BiLSTM/PhoBERT")
    args = parser.parse_args()

    t0 = time.time()
    stages = ["A", "B", "C"] if args.stage == "all" else [args.stage]
    for s in stages:
        if s == "A":
            stage_a(force=args.force)
        elif s == "B":
            stage_b(skip_retrain=args.skip_retrain, device=args.device, epochs=args.epochs)
        elif s == "C":
            stage_c(skip_retrain=args.skip_retrain, device=args.device, epochs=args.epochs)

    log.info("=" * 70)
    log.info("Phase 0 COMPLETE in %.1fs", time.time() - t0)
    log.info("Verify with:  python -m pytest tests/test_save_load.py -v")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
