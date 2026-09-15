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

__all__ = [
    "TfidfFeatureExtractor",
    "extract_tfidf_features",
]


class TfidfFeatureExtractor:
    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Optional[Tuple[int, int]] = None,
        min_df: Optional[int] = None,
        max_df: Optional[float] = None
    ):
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
            strip_accents=None,
            lowercase=True
        )
        
        self.is_fitted = False
    
    def fit(self, texts: pd.Series) -> 'TfidfFeatureExtractor':
        log.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents...")
        self.vectorizer.fit(texts.astype(str))
        self.is_fitted = True
        
        vocab_size = len(self.vectorizer.vocabulary_)
        log.info(f"Vocabulary size: {vocab_size}")
        
        return self
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        features = self.vectorizer.transform(texts.astype(str))
        log.info(f"Transformed {len(texts)} documents to shape {features.shape}")
        
        return features
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> list:
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: str) -> None:
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
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    log.info("Loading datasets...")
    train_df = load_csv(train_path, required_columns=['text', 'label'])
    val_df = load_csv(val_path, required_columns=['text', 'label'])
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    log.info(f"Train: {len(train_df)} samples")
    log.info(f"Val: {len(val_df)} samples")
    log.info(f"Test: {len(test_df)} samples")
    
    extractor = TfidfFeatureExtractor(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    log.info("\nExtracting TF-IDF features...")
    X_train = extractor.fit_transform(train_df['text'])
    X_val = extractor.transform(val_df['text'])
    X_test = extractor.transform(test_df['text'])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    extractor.save(vectorizer_path)
    
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
