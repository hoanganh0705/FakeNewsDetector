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

MODEL_DIR_MAP: Dict[str, str] = {
    'Logistic Regression': 'lr',
    'SVM': 'svm',
    'BiLSTM': 'bilstm',
    'PhoBERT': 'bert',
}

DIR_MODEL_MAP: Dict[str, str] = {v: k for k, v in MODEL_DIR_MAP.items()}



def load_metrics(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_all_metrics(
    experiments_dir: Optional[str] = None,
    models: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:   
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


def compute_balanced_class_weights(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_class_weight
    return compute_class_weight('balanced', classes=np.unique(y), y=y)

def set_reproducibility_seeds(seed: Optional[int] = None) -> None:
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


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    source_path: str = "<unknown>",
) -> None:
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, **read_csv_kwargs)
    if required_columns is not None:
        validate_dataframe_columns(df, required_columns, source_path=path)
    return df


def _git_rev() -> Optional[str]:
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
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class ExperimentTracker:
    LOG_FILE = "experiment_log.json"

    def __init__(self, experiment_dir: str) -> None:
        self.experiment_dir = experiment_dir
        self.log_path = os.path.join(experiment_dir, self.LOG_FILE)

    def log_run(
        self,
        model_name: str,
        config: dict,
        metrics: dict,
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
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
    
    def _load(self) -> list:
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as fh:
                return json.load(fh)
        return []

    def _save(self, records: list) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        with open(self.log_path, "w") as fh:
            json.dump(records, fh, indent=2, default=str)
