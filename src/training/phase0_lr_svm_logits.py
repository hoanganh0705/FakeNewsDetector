"""
Phase 0 — Stage A helper: capture raw_logits.pkl for LR + SVM.

This script is the **no-network, no-retrain** path through Phase 0.
It loads the existing `lr_model.pkl` and `svm_model.pkl` checkpoints
(which are already present in `experiments/{lr,svm}/`) and re-runs
inference on the val + test splits to produce `raw_logits.pkl` —
needed by Phase 3 (Knowledge Distillation) and as a sanity check
that we can faithfully reproduce the paper's reported LR/SVM numbers.

Outputs
-------
- experiments/lr/raw_logits.pkl
    dict with keys:
        y_true, y_pred, y_prob, raw_logit, split
- experiments/svm/raw_logits.pkl
    same shape

`raw_logit` is the **pre-sigmoid** (LR) / **pre-platt** (SVM) score.
For LR:    `model.decision_function(X)` (logit, before sigmoid)
For SVM:   when `use_linear=True` the wrapped estimator is a
           `CalibratedClassifierCV(LinearSVC)` — we unwrap and call
           `calibrated_classifier.base_estimator.decision_function(X)`
           to get the underlying LinearSVC's `decision_function`.
           When `use_linear=False`, the model is a kernel SVC with
           `probability=True` — we use `model.decision_function(X)`
           (still meaningful, even if not strictly a logit).

Run
---
    python src/training/phase0_lr_svm_logits.py

This script is **idempotent** — it always overwrites `raw_logits.pkl`.
No training, no network access required.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np

# Make `config` importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _to_list(arr):
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _resolve_raw_logit(model, X: np.ndarray) -> np.ndarray:
    """Return the *pre-sigmoid* score (or best available surrogate).

    For LR: returns `decision_function(X)` — exactly the logit.
    For SVM:
      - LinearSVC wrapped by CalibratedClassifierCV → unwrap, call base
        LinearSVC.decision_function (logits).
      - Kernel SVC (probability=True) → `decision_function(X)` (signed
        distance; equivalent role to logit for distillation).
    """
    # CalibratedClassifierCV path
    if hasattr(model, "calibrated_classifiers_") and hasattr(model, "estimator"):
        # sklearn ≥1.8 uses `.estimator`; older versions use `.base_estimator`.
        base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        if base is None:
            raise RuntimeError(
                "CalibratedClassifierCV present but cannot resolve base estimator."
            )
        if hasattr(base, "decision_function"):
            return np.asarray(base.decision_function(X), dtype=np.float64)
        raise RuntimeError(
            f"Base estimator {type(base).__name__} has no decision_function."
        )

    # Native decision_function (LR, plain SVC)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)

    raise RuntimeError(
        f"Cannot extract raw logits from model of type {type(model).__name__}"
    )


def _run_one(
    *,
    model_name: str,
    model_path: Path,
    model_dir: Path,
    features_path: Path,
) -> dict:
    """Load model + features, run on val+test, persist raw_logits.pkl."""
    log.info("=" * 60)
    log.info("Phase 0 / %s", model_name)
    log.info("=" * 60)

    log.info("Loading features from %s ...", features_path)
    feats = joblib.load(features_path)

    X_train, y_train = feats["X_train"], feats["y_train"]
    X_val, y_val = feats["X_val"], feats["y_val"]
    X_test, y_test = feats["X_test"], feats["y_test"]

    log.info("Loading model from %s ...", model_path)
    bundle = joblib.load(model_path)
    # All trainers store a dict under `lr_model.pkl`/`svm_model.pkl` with
    # `model` key pointing to the actual sklearn estimator.
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
    else:
        # Fallback for plain estimator pickles (defensive)
        model = bundle

    results = {}

    # Build a dict with both splits in one file (one file per model = 1
    # artefact for Phase 3 KD).  Keys are 'val' and 'test'.
    raw_logits_combined = {"model_name": model_name}

    for split_name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        t0 = time.time()
        y_pred = np.asarray(model.predict(X))
        # All sklearn classifiers expose predict_proba — both LR (sigmoid
        # of decision_function) and SVM (Platt-calibrated) produce P(y=1).
        y_prob = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
        raw_logit = _resolve_raw_logit(model, X)
        elapsed = time.time() - t0

        raw_logits_combined[split_name] = {
            "y_true": _to_list(y),
            "y_pred": _to_list(y_pred),
            "y_prob": _to_list(y_prob),
            "raw_logit": _to_list(raw_logit),
            "n_samples": int(len(y)),
        }
        log.info(
            "[%s] %s — n=%d  pred-prob[0:3]=%s  raw_logit[0:3]=%s  (%.2fs)",
            model_name, split_name, len(y),
            np.round(y_prob[:3], 4).tolist(),
            np.round(raw_logit[:3], 4).tolist(),
            elapsed,
        )

        metrics = compute_metrics(np.asarray(y), y_pred, y_prob)
        results[split_name] = {
            "n_samples": int(len(y)),
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "f1_real": float(metrics.get("f1_real", float("nan"))),
            "f1_fake": float(metrics.get("f1_fake", float("nan"))),
        }

    # Persist single file with both splits
    out_path = model_dir / "raw_logits.pkl"
    joblib.dump(raw_logits_combined, out_path)
    log.info("Saved %s", out_path)

    # Pretty-print metrics summary
    summary_path = model_dir / "phase0_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model_name, **results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
            f,
            indent=2,
            ensure_ascii=False,
        )
    log.info("%s metrics: %s", model_name, json.dumps(results, indent=2))
    test_metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    print_metrics(test_metrics, title=f"{model_name} — Test Set")

    return results


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Phase 0 / Stage A — capturing raw_logits for LR + SVM")
    log.info("This is the no-network / no-retrain path.")

    # ── LR ──────────────────────────────────────────────────────────
    _run_one(
        model_name="Logistic Regression",
        model_path=Path(cfg.PATHS.lr_dir) / "lr_model.pkl",
        model_dir=Path(cfg.PATHS.lr_dir),
        features_path=Path(cfg.PATHS.tfidf_dir) / "tfidf_features.pkl",
    )

    # ── SVM ─────────────────────────────────────────────────────────
    _run_one(
        model_name="SVM",
        model_path=Path(cfg.PATHS.svm_dir) / "svm_model.pkl",
        model_dir=Path(cfg.PATHS.svm_dir),
        features_path=Path(cfg.PATHS.tfidf_dir) / "tfidf_features.pkl",
    )

    log.info("=" * 60)
    log.info("Phase 0 / Stage A COMPLETE")
    log.info("Outputs:")
    log.info("  experiments/lr/raw_logits.pkl")
    log.info("  experiments/svm/raw_logits.pkl")
    log.info("  experiments/lr/phase0_summary.json")
    log.info("  experiments/svm/phase0_summary.json")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
