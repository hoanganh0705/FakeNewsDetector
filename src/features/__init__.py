# Feature engineering module
"""
Feature extraction for different model types:
- TF-IDF: For traditional ML models (Logistic Regression, SVM)
- Word Embeddings: For BiLSTM
- PhoBERT Tokenizer: For PhoBERT transformer model
"""

from .tfidf_features import TfidfFeatureExtractor
from .embedding_features import EmbeddingFeatureExtractor
from .phobert_features import PhoBertFeatureExtractor

__all__ = [
    'TfidfFeatureExtractor',
    'EmbeddingFeatureExtractor', 
    'PhoBertFeatureExtractor'
]
