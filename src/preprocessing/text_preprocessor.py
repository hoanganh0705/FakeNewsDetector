"""
Text preprocessing module for Vietnamese fake news detection.
Handles text cleaning and normalization.

Vectorization has been consolidated into ``src.features.tfidf_features.TfidfFeatureExtractor``
to avoid duplication.  This module focuses exclusively on text-level cleaning.
"""

import re
from typing import List
import pandas as pd

from src.utils.logger import get_logger
from config import cfg

_log = get_logger(__name__)


class TextPreprocessor:
    """
    Text preprocessing pipeline for Vietnamese text.
    Provides cleaning / normalization utilities only.

    For TF-IDF vectorization use :class:`src.features.tfidf_features.TfidfFeatureExtractor`.
    """

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
        
        # Remove markdown-style URL patterns like [<URL>](<URL>) or [text](https://...)
        text = re.sub(r'\[.*?\]\([^)]*\)', '', text)

        # Remove bare URLs and <URL> placeholder tags
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
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


def clean_dataset(
    df: pd.DataFrame,
    text_col: str = 'text',
    date_col: str = 'date',
    min_words: int = None
) -> pd.DataFrame:
    """
    Apply full dataset-level quality fixes based on the dataset quality report.

    Steps applied (in order):
      1. Deduplicate on text column — prevents data leakage across splits.
      2. Drop very-short / broken records (< min_words valid words).
      3. Standardize date column to ISO 8601 (YYYY-MM-DD).
      4. Re-number the 'id' column sequentially.

    Args:
        df:        DataFrame loaded from raw.csv (or any raw CSV).
        text_col:  Name of the text column.
        date_col:  Name of the date column.
        min_words: Minimum number of whitespace-separated words required.

    Returns:
        Cleaned DataFrame with reset index.
    """
    min_words = min_words if min_words is not None else cfg.DATA.min_word_count

    original_len = len(df)

    # 1. Deduplicate
    df = df.drop_duplicates(subset=text_col).reset_index(drop=True)
    after_dedup = len(df)
    _log.info("Deduplication: %d → %d rows (removed %d duplicates)",
              original_len, after_dedup, original_len - after_dedup)

    # 2. Drop very-short / broken records
    df = df[df[text_col].fillna('').astype(str).str.split().str.len() >= min_words]
    df = df.reset_index(drop=True)
    after_short = len(df)
    _log.info("Short-record filter (<%d words): %d → %d rows (removed %d records)",
              min_words, after_dedup, after_short, after_dedup - after_short)

    # 3. Standardize date column to YYYY-MM-DD
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
        _log.info("Date column '%s' standardized to ISO 8601.", date_col)

    # 4. Re-number IDs
    if 'id' in df.columns:
        df['id'] = range(1, len(df) + 1)

    _log.info("Dataset cleaning complete. Final size: %d records.", len(df))
    return df


def load_data(
    data_path: str,
    text_col: str = 'text',
    label_col: str = 'label',
    apply_cleaning: bool = True
):
    """
    Load data from CSV file with optional dataset-level cleaning.

    Args:
        data_path:      Path to CSV file (e.g. data/raw/raw.csv).
        text_col:       Name of text column.
        label_col:      Name of label column.
        apply_cleaning: If True, run clean_dataset() before returning.

    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(data_path)
    _log.info("Loaded %d records from %s", len(df), data_path)

    if apply_cleaning:
        df = clean_dataset(df, text_col=text_col, min_words=cfg.DATA.min_word_count)

    texts = df[text_col].fillna('').astype(str).tolist()
    labels = df[label_col].values
    return texts, labels
