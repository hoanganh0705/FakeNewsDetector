"""
TF-IDF Feature Extraction for Traditional ML Models (Logistic Regression, SVM)


TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical
vectors based on word importance across the corpus.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from typing import Tuple, Optional
from config import cfg
from src.utils.common import load_csv

from src.utils.logger import get_logger
log = get_logger(__name__)


class TfidfFeatureExtractor:
    """
    TF-IDF feature extractor for Vietnamese fake news detection.
    
    Attributes:
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
        min_df: Minimum document frequency for a term to be included
        max_df: Maximum document frequency for a term to be included
    """
    
    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Optional[Tuple[int, int]] = None,
        min_df: Optional[int] = None,
        max_df: Optional[float] = None
    ):
        """
        Initialize the TF-IDF feature extractor.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: (min_n, max_n) for n-gram extraction
            min_df: Ignore terms with document frequency below this threshold
            max_df: Ignore terms with document frequency above this threshold
        """
        # Use central config defaults when not provided
        self.max_features = max_features if max_features is not None else cfg.TFIDF.max_features
        self.ngram_range = ngram_range if ngram_range is not None else cfg.TFIDF.ngram_range
        self.min_df = min_df if min_df is not None else cfg.TFIDF.min_df
        self.max_df = max_df if max_df is not None else cfg.TFIDF.max_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=cfg.TFIDF.sublinear_tf,
            strip_accents=None,  # Keep Vietnamese accents
            lowercase=True
        )
        
        self.is_fitted = False
    
    def fit(self, texts: pd.Series) -> 'TfidfFeatureExtractor':
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts: Series of text documents
            
        Returns:
            self
        """
        log.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents...")
        self.vectorizer.fit(texts.astype(str))
        self.is_fitted = True
        
        vocab_size = len(self.vectorizer.vocabulary_)
        log.info(f"Vocabulary size: {vocab_size}")
        
        return self
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts to TF-IDF feature matrix.
        
        Args:
            texts: Series of text documents
            
        Returns:
            TF-IDF feature matrix (sparse or dense)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        features = self.vectorizer.transform(texts.astype(str))
        log.info(f"Transformed {len(texts)} documents to shape {features.shape}")
        
        return features
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: Series of text documents
            
        Returns:
            TF-IDF feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> list:
        """Get the feature names (vocabulary terms)."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: str) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            path: Path to save the vectorizer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'is_fitted': self.is_fitted
        }, path)
        log.info(f"Saved TF-IDF vectorizer to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TfidfFeatureExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            path: Path to the saved vectorizer
            
        Returns:
            Loaded TfidfFeatureExtractor instance
        """
        data = joblib.load(path)
        
        extractor = cls(
            max_features=data['max_features'],
            ngram_range=data['ngram_range'],
            min_df=data['min_df'],
            max_df=data['max_df']
        )
        extractor.vectorizer = data['vectorizer']
        extractor.is_fitted = data['is_fitted']
        
        log.info(f"Loaded TF-IDF vectorizer from {path}")
        return extractor


def extract_tfidf_features(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    max_features: Optional[int] = None,
    ngram_range: Optional[Tuple[int, int]] = None
) -> dict:
    """
    Extract TF-IDF features from train/val/test datasets.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        output_dir: Directory to save features
        max_features: Maximum vocabulary size
        ngram_range: N-gram range
        
    Returns:
        Dictionary with feature matrices and labels
    """


    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    log.info("Loading datasets...")
    train_df = load_csv(train_path, required_columns=['text', 'label'])
    val_df = load_csv(val_path, required_columns=['text', 'label'])
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    log.info(f"Train: {len(train_df)} samples")
    log.info(f"Val: {len(val_df)} samples")
    log.info(f"Test: {len(test_df)} samples")
    
    # Initialize and fit TF-IDF extractor on training data
    extractor = TfidfFeatureExtractor(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    # Extract features
    log.info("\nExtracting TF-IDF features...")
    X_train = extractor.fit_transform(train_df['text'])
    X_val = extractor.transform(val_df['text'])
    X_test = extractor.transform(test_df['text'])
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    extractor.save(vectorizer_path)
    
    # Save features
    features_path = os.path.join(output_dir, 'tfidf_features.pkl')
    joblib.dump({
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }, features_path)
    log.info(f"Saved features to {features_path}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'extractor': extractor
    }


if __name__ == "__main__":
    # Example usage
    features = extract_tfidf_features(
        train_path=os.path.join(cfg.PATHS.splits_dir, 'train.csv'),
        val_path=os.path.join(cfg.PATHS.splits_dir, 'val.csv'),
        test_path=os.path.join(cfg.PATHS.splits_dir, 'test.csv'),
        output_dir=cfg.PATHS.tfidf_dir,
        max_features=cfg.TFIDF.max_features,
        ngram_range=cfg.TFIDF.ngram_range
    )
    
    print("\n" + "="*50)
    print("TF-IDF Feature Extraction Complete!")
    print("="*50)
    print(f"Train features shape: {features['X_train'].shape}")
    print(f"Val features shape: {features['X_val'].shape}")
    print(f"Test features shape: {features['X_test'].shape}")
