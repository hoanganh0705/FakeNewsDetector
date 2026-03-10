"""
Master training script to train all models sequentially.

Usage:
    python src/training/train_all.py
"""

import importlib
import time
import traceback
from datetime import datetime

from src.utils.logger import get_logger
log = get_logger(__name__)

# (display name, module path)
_MODELS = [
    ('Logistic Regression', 'src.training.train_lr'),
    ('SVM',                 'src.training.train_svm'),
    ('BiLSTM',              'src.training.train_bilstm'),
    ('PhoBERT',             'src.training.train_phobert'),
]


def main():
    """Train all models sequentially."""

    log.info("=" * 60)
    log.info("TRAINING ALL MODELS")
    log.info("=" * 60)
    log.info("Started at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    results = {}
    total_start = time.time()

    for idx, (name, module_path) in enumerate(_MODELS, 1):
        log.info("")
        log.info("=" * 60)
        log.info("Model %d/%d: %s", idx, len(_MODELS), name)
        log.info("=" * 60)

        try:
            start = time.time()
            mod = importlib.import_module(module_path)
            _trainer, metrics = mod.main()
            results[name] = {
                'status': 'success',
                'test_accuracy': metrics['accuracy'],
                'test_f1': metrics['f1_macro'],
                'time': time.time() - start,
            }
        except Exception as e:
            log.error("Error training %s: %s", name, e)
            traceback.print_exc()
            results[name] = {'status': 'failed', 'error': str(e)}

    # ── Summary ──────────────────────────────────────────────────
    total_time = time.time() - total_start

    log.info("")
    log.info("=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    log.info("%-25s %-10s %-12s %-12s %-10s", 'Model', 'Status', 'Accuracy', 'F1-Score', 'Time')
    log.info("-" * 70)

    for model, result in results.items():
        if result['status'] == 'success':
            log.info("%-25s %-10s %-12.4f %-12.4f %.1fs",
                     model, 'OK', result['test_accuracy'], result['test_f1'], result['time'])
        else:
            log.info("%-25s %-10s %-12s %-12s %-10s", model, 'FAIL', 'N/A', 'N/A', 'N/A')

    log.info("-" * 70)
    log.info("Total training time: %.1f minutes", total_time / 60)
    log.info("Finished at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Find best model
    best_model = max(
        (m for m, r in results.items() if r['status'] == 'success'),
        key=lambda m: results[m]['test_f1'],
        default=None,
    )

    if best_model:
        log.info("Best Model: %s (F1: %.4f)", best_model, results[best_model]['test_f1'])

    log.info("=" * 60)
    log.info("ALL TRAINING COMPLETE!")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
