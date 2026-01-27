"""
Text preprocessing module for Vietnamese fake news detection.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import string
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import os


class TextPreprocessor:
    """
    Text preprocessing pipeline for Vietnamese text.
    Includes cleaning, normalization, and vectorization.
    """
    
    def __init__(
        self,
        vectorizer_type: str = 'tfidf',
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            vectorizer_type: 'tfidf' or 'count'
            max_features: Maximum number of features
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df
            )
        
        self.is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Vietnamese text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<URL>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove emojis and special unicode characters
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]
    
    def fit(self, texts: List[str]):
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts: List of training texts
        """
        cleaned_texts = self.preprocess_texts(texts)
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        cleaned_texts = self.preprocess_texts(texts)
        return self.vectorizer.transform(cleaned_texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts in one step.
        
        Args:
            texts: List of training texts
            
        Returns:
            Feature matrix
        """
        cleaned_texts = self.preprocess_texts(texts)
        self.is_fitted = True
        return self.vectorizer.fit_transform(cleaned_texts)
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names from the vectorizer."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: str):
        """Save the preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'vectorizer_type': self.vectorizer_type,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TextPreprocessor':
        """Load a preprocessor from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(
            vectorizer_type=data['vectorizer_type'],
            max_features=data['max_features'],
            ngram_range=data['ngram_range'],
            min_df=data['min_df'],
            max_df=data['max_df']
        )
        preprocessor.vectorizer = data['vectorizer']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor


def load_data(data_path: str, text_col: str = 'text', label_col: str = 'label'):
    """
    Load data from CSV file.
    
    Args:
        data_path: Path to CSV file
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(data_path)
    texts = df[text_col].fillna('').astype(str).tolist()
    labels = df[label_col].values
    return texts, labels
