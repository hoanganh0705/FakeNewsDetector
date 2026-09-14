"""
``src.models`` — model wrappers for FakeNewsDetector.

Re-exports the three model classes so that trainers and evaluation scripts
can import from a single place::

    from src.models import BiLSTMClassifier, PhoBertClassifier, StudentBiLSTM

"""

from src.models.bilstm_model import BiLSTMClassifier
from src.models.phobert_model import PhoBertClassifier
from src.models.student_model import StudentBiLSTM

__all__ = [
    "BiLSTMClassifier",
    "PhoBertClassifier",
    "StudentBiLSTM",
]
