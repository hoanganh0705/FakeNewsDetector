# Training module
"""
Training scripts for all models:
- Logistic Regression (train_lr.py)
- SVM (train_svm.py)
- BiLSTM (train_bilstm.py)
- PhoBERT (train_phobert.py)
- Train all models (train_all.py)
"""

from src.training.runner import save_training_results
from src.training.train_lr import LogisticRegressionTrainer
from src.training.train_svm import SVMTrainer
from src.training.train_bilstm import BiLSTMTrainer
from src.training.train_phobert import PhoBertTrainer
from src.training.train_student import StudentBiLSTMTrainer, distillation_loss
from src.training.reproduce_predictions import (
    _resolve_raw_logit_sklearn,
    stage_a,
    stage_b,
    stage_c,
)
from src.training.distillation_evaluation import (
    _infer_teacher_params,
    _count_params_torch,
    _size_mb_torch,
    _measure_inference_time,
    run_distillation_evaluation,
)

__all__ = [
    "save_training_results",
    "LogisticRegressionTrainer",
    "SVMTrainer",
    "BiLSTMTrainer",
    "PhoBertTrainer",
    "StudentBiLSTMTrainer",
    "distillation_loss",
    "_resolve_raw_logit_sklearn",
    "stage_a",
    "stage_b",
    "stage_c",
    "_infer_teacher_params",
    "_count_params_torch",
    "_size_mb_torch",
    "_measure_inference_time",
    "run_distillation_evaluation",
]
