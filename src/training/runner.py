"""
Shared post-training save/log helpers for all training scripts.

Eliminates the boilerplate that was copy-pasted across train_lr, train_svm,
train_bilstm, and train_phobert.
"""

import os
import joblib
from datetime import datetime

from src.evaluation.metrics import save_metrics
from src.utils.common import ExperimentTracker
from src.utils.logger import get_logger

log = get_logger(__name__)


def _to_list(arr):
    """Convert an array-like to a plain Python list."""
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return list(arr)


def save_training_results(
    *,
    model_name: str,
    model_dir: str,
    model_path: str,
    metrics_dict: dict,
    test_metrics: dict,
    y_true,
    y_pred,
    y_prob,
    experiment_config: dict,
) -> dict:
    """
    Save metrics, predictions, and experiment log after training.

    This consolidates the post-training boilerplate that was duplicated
    across every ``train_*.py`` script.

    Args:
        model_name: Human-readable model name (e.g. 'Logistic Regression').
        model_dir: Directory where artefacts are saved.
        model_path: Path to the saved model file.
        metrics_dict: Full metrics dict to persist as ``metrics.json``.
        test_metrics: Test-set metrics dict for the experiment tracker.
        y_true: Ground-truth labels for the test set.
        y_pred: Predicted labels for the test set.
        y_prob: Predicted probabilities for the test set.
        experiment_config: Config dict logged by ``ExperimentTracker``.

    Returns:
        dict with ``metrics_path`` and ``predictions_path``.
    """
    # Auto-add timestamp if the caller didn't
    metrics_dict.setdefault('timestamp', datetime.now().isoformat())

    metrics_path = os.path.join(model_dir, 'metrics.json')
    save_metrics(metrics_dict, metrics_path)

    predictions_path = os.path.join(model_dir, 'predictions.pkl')
    joblib.dump(
        {
            'y_true': _to_list(y_true),
            'y_pred': _to_list(y_pred),
            'y_prob': _to_list(y_prob),
        },
        predictions_path,
    )

    ExperimentTracker(model_dir).log_run(
        model_name=model_name,
        config=experiment_config,
        metrics=test_metrics,
    )

    log.info("=" * 60)
    log.info("TRAINING COMPLETE!")
    log.info("=" * 60)
    log.info("Model saved to: %s", model_path)
    log.info("Metrics saved to: %s", metrics_path)
    log.info("Predictions saved to: %s", predictions_path)

    return {'metrics_path': metrics_path, 'predictions_path': predictions_path}
