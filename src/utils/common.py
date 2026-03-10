"""
Shared utility helpers used across analysis, evaluation, and training scripts.

Consolidates duplicated logic that was previously copy-pasted into many files:
- ``load_metrics`` / ``load_all_metrics`` — loading experiment JSON metrics
- ``MODEL_DIR_MAP`` — canonical model-name → experiment-directory mapping
- ``compute_balanced_class_weights`` — sklearn balanced class weights
- ``load_csv`` — ``pd.read_csv`` wrapper with column validation
- ``ExperimentTracker`` — lightweight experiment versioning
"""

import json
import os
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import cfg

# ──────────────────────────────────────────────────────────────────────
# Canonical mapping from human-readable model name to experiment dir name.
# Import this instead of re-defining the dict in every script.
# ──────────────────────────────────────────────────────────────────────
MODEL_DIR_MAP: Dict[str, str] = {
    'Logistic Regression': 'lr',
    'SVM': 'svm',
    'BiLSTM': 'bilstm',
    'PhoBERT': 'bert',
}

# Reverse mapping (dir name → display name), occasionally useful.
DIR_MODEL_MAP: Dict[str, str] = {v: k for k, v in MODEL_DIR_MAP.items()}


# ──────────────────────────────────────────────────────────────────────
# Metrics I/O
# ──────────────────────────────────────────────────────────────────────

def load_metrics(path: str) -> dict:
    """Load a single ``metrics.json`` file and return its contents."""
    with open(path, 'r') as f:
        return json.load(f)


def load_all_metrics(
    experiments_dir: Optional[str] = None,
    models: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """Load ``metrics.json`` for every model that has one.

    Args:
        experiments_dir: Root experiments dir.  Defaults to ``cfg.PATHS.experiments_dir``.
        models:          ``{display_name: dir_name}`` mapping.  Defaults to ``MODEL_DIR_MAP``.

    Returns:
        ``{model_display_name: metrics_dict}`` for each model that has a metrics file.
    """
    if experiments_dir is None:
        experiments_dir = cfg.PATHS.experiments_dir
    if models is None:
        models = MODEL_DIR_MAP

    metrics: Dict[str, dict] = {}
    for name, dir_name in models.items():
        path = os.path.join(experiments_dir, dir_name, 'metrics.json')
        if os.path.exists(path):
            metrics[name] = load_metrics(path)
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Class-weight computation
# ──────────────────────────────────────────────────────────────────────

def compute_balanced_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced class weights for a label array.

    Wraps ``sklearn.utils.class_weight.compute_class_weight`` so that
    training scripts don't each have to do an inline import.

    Args:
        y:  1-D array of integer class labels.

    Returns:
        Array of shape ``(n_classes,)`` with balanced weights.
    """
    from sklearn.utils.class_weight import compute_class_weight
    return compute_class_weight('balanced', classes=np.unique(y), y=y)


# ──────────────────────────────────────────────────────────────────────
# Reproducibility seeds for PyTorch training
# ──────────────────────────────────────────────────────────────────────

def set_reproducibility_seeds(seed: Optional[int] = None) -> None:
    """Set random seeds for NumPy & PyTorch to ensure reproducible training.

    Enables ``torch.backends.cudnn.deterministic`` and disables benchmark
    mode so that cuDNN selects the same algorithms every run.

    Args:
        seed: Explicit seed value.  Falls back to ``cfg.RANDOM_STATE``.
    """
    import random
    import torch

    seed = seed if seed is not None else cfg.RANDOM_STATE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# CSV loading with column validation
# ──────────────────────────────────────────────────────────────────────

def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    source_path: str = "<unknown>",
) -> None:
    """Raise a descriptive error if *df* is missing any required columns.

    Args:
        df:               The :class:`~pandas.DataFrame` to check.
        required_columns: Column names that must be present.
        source_path:      File path shown in the error message for easier debugging.

    Raises:
        ValueError: When one or more required columns are absent.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV '{source_path}' is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def load_csv(
    path: str,
    required_columns: Optional[Sequence[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV file with optional column validation.

    A thin wrapper around :func:`pandas.read_csv` that adds:
    * A human-friendly :class:`FileNotFoundError` when the file is missing.
    * Automatic column-existence checks via *required_columns*.

    Args:
        path:             Path to the CSV file.
        required_columns: If given, the loaded DataFrame is validated to contain
                          all of these columns.  ``None`` skips the check.
        **read_csv_kwargs: Forwarded to :func:`pandas.read_csv`.

    Returns:
        The loaded :class:`~pandas.DataFrame`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, **read_csv_kwargs)
    if required_columns is not None:
        validate_dataframe_columns(df, required_columns, source_path=path)
    return df


# ──────────────────────────────────────────────────────────────────────
# Lightweight experiment versioning
# ──────────────────────────────────────────────────────────────────────

def _git_rev() -> Optional[str]:
    """Return the short git commit hash, or *None* if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _config_hash(config_dict: dict) -> str:
    """Deterministic SHA-256 fingerprint (first 12 hex chars) of *config_dict*."""
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class ExperimentTracker:
    """Append-only ``experiment_log.json`` inside each model's experiment dir.

    Each training run appends a record with timestamp, git commit, config
    hash, and selected metrics so that earlier runs are never lost.

    Usage in a training script::

        tracker = ExperimentTracker(model_dir)
        tracker.log_run(
            model_name="Logistic Regression",
            config={"C": 1.0, "max_iter": 2000},
            metrics=test_metrics,
        )
    """

    LOG_FILE = "experiment_log.json"

    def __init__(self, experiment_dir: str) -> None:
        self.experiment_dir = experiment_dir
        self.log_path = os.path.join(experiment_dir, self.LOG_FILE)

    # ── public API ─────────────────────────────────────────────────────

    def log_run(
        self,
        model_name: str,
        config: dict,
        metrics: dict,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Append a new run record and return it.

        Args:
            model_name: Human-readable model label.
            config:     Hyperparameter dict used for this run.
            metrics:    Evaluation metrics dict from ``compute_metrics``.
            extra:      Any additional key/value pairs to store.

        Returns:
            The run record that was appended.
        """
        record = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "git_commit": _git_rev(),
            "config_hash": _config_hash(config),
            "model": model_name,
            "config": config,
            "metrics_summary": {
                k: metrics[k]
                for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro")
                if k in metrics
            },
        }
        if extra:
            record["extra"] = extra

        history = self._load()
        history.append(record)
        self._save(history)
        return record

    # ── internals ──────────────────────────────────────────────────────

    def _load(self) -> list:
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as fh:
                return json.load(fh)
        return []

    def _save(self, records: list) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        with open(self.log_path, "w") as fh:
            json.dump(records, fh, indent=2, default=str)
