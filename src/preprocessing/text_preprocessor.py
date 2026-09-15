import re
from typing import List
import pandas as pd

from src.utils.logger import get_logger
from config import cfg

_log = get_logger(__name__)


class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        text = re.sub(r'\[.*?\]\([^)]*\)', '', text)

        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<URL>', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'<[^>]+>', '', text)
        
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        return [self.clean_text(text) for text in texts]


def clean_dataset(
    df: pd.DataFrame,
    text_col: str = 'text',
    date_col: str = 'date',
    min_words: int = None
) -> pd.DataFrame:
    min_words = min_words if min_words is not None else cfg.DATA.min_word_count

    original_len = len(df)

    df = df.drop_duplicates(subset=text_col).reset_index(drop=True)
    after_dedup = len(df)
    _log.info("Deduplication: %d → %d rows (removed %d duplicates)",
              original_len, after_dedup, original_len - after_dedup)

    df = df[df[text_col].fillna('').astype(str).str.split().str.len() >= min_words]
    df = df.reset_index(drop=True)
    after_short = len(df)
    _log.info("Short-record filter (<%d words): %d → %d rows (removed %d records)",
              min_words, after_dedup, after_short, after_dedup - after_short)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
        _log.info("Date column '%s' standardized to ISO 8601.", date_col)

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
    df = pd.read_csv(data_path)
    _log.info("Loaded %d records from %s", len(df), data_path)

    if apply_cleaning:
        df = clean_dataset(df, text_col=text_col, min_words=cfg.DATA.min_word_count)

    texts = df[text_col].fillna('').astype(str).tolist()
    labels = df[label_col].values
    return texts, labels
