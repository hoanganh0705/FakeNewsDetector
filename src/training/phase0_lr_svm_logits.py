from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

log = get_logger(__name__)


def _to_list(arr):
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _resolve_raw_logit(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "calibrated_classifiers_") and hasattr(model, "estimator"):
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
    log.info("=" * 60)
    log.info("=" * 60)

    log.info("Loading features from %s ...", features_path)
    feats = joblib.load(features_path)

    X_train, y_train = feats["X_train"], feats["y_train"]
    X_val, y_val = feats["X_val"], feats["y_val"]
    X_test, y_test = feats["X_test"], feats["y_test"]

    log.info("Loading model from %s ...", model_path)
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
    else:
        model = bundle

    results = {}

    raw_logits_combined = {"model_name": model_name}

    for split_name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        t0 = time.time()
        y_pred = np.asarray(model.predict(X))
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

    out_path = model_dir / "raw_logits.pkl"
    joblib.dump(raw_logits_combined, out_path)
    log.info("Saved %s", out_path)

    summary_path = model_dir / "summary.json"
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


def main() -> None:
    _run_one(
        model_name="Logistic Regression",
        model_path=Path(cfg.PATHS.lr_dir) / "lr_model.pkl",
        model_dir=Path(cfg.PATHS.lr_dir),
        features_path=Path(cfg.PATHS.tfidf_dir) / "tfidf_features.pkl",
    )

    _run_one(
        model_name="SVM",
        model_path=Path(cfg.PATHS.svm_dir) / "svm_model.pkl",
        model_dir=Path(cfg.PATHS.svm_dir),
        features_path=Path(cfg.PATHS.tfidf_dir) / "tfidf_features.pkl",
    )


if __name__ == "__main__":
    main()
