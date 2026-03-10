"""
PhoBERT Feature Extraction for Transformer Model


PhoBERT is a pre-trained language model for Vietnamese, based on RoBERTa architecture.
This module provides tokenization and dataset creation for PhoBERT fine-tuning.
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import cfg
from src.utils.common import load_csv

from src.utils.logger import get_logger
log = get_logger(__name__)


class PhoBertFeatureExtractor:
    """
    PhoBERT tokenizer wrapper for Vietnamese text classification.
    
    Uses tokenizer defined in `cfg.PHOBERT.model_name` to convert text to input IDs and attention masks.
    """
    
    # Use centralized config for model name and max length
    MODEL_NAME = cfg.PHOBERT.model_name

    def __init__(self, max_length: int = None):
        """
        Initialize PhoBERT tokenizer.

        Args:
            max_length: Maximum sequence length; defaults to `cfg.PHOBERT.max_seq_len`
        """
        self.max_length = int(max_length or cfg.PHOBERT.max_seq_len)

        # Try local cache first, then fall back to HuggingFace Hub download
        local_cache_dir = os.path.join(cfg.PATHS.features_dir, 'phobert_tokenizer_cache')
        log.info(f"Loading PhoBERT tokenizer ({self.MODEL_NAME})...")
        try:
            if os.path.isdir(local_cache_dir) and os.listdir(local_cache_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(local_cache_dir)
                log.info("Tokenizer loaded from local cache.")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                os.makedirs(local_cache_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_cache_dir)
                log.info("Tokenizer downloaded and cached locally.")
        except OSError:
            # Offline or air-gapped: try local cache as last resort
            if os.path.isdir(local_cache_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(local_cache_dir)
                log.info("Tokenizer loaded from local cache (offline fallback).")
            else:
                raise
        log.info(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def tokenize(
        self, 
        texts: pd.Series,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts using PhoBERT tokenizer.
        
        Args:
            texts: Series of text documents
            return_tensors: Whether to return PyTorch tensors
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        log.info(f"Tokenizing {len(texts)} documents...")
        
        # Convert to list of strings
        text_list = texts.astype(str).tolist()
        
        # Tokenize
        encoded = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt' if return_tensors else None
        )
        
        log.info(f"Tokenized to shape: {encoded['input_ids'].shape}")
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def save_config(self, path: str) -> None:
        """Save configuration (tokenizer is loaded from HuggingFace)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model_name': self.MODEL_NAME,
            'max_length': self.max_length
        }, path)
        log.info(f"Saved config to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PhoBertFeatureExtractor':
        """Load extractor from config."""
        config = joblib.load(path)
        return cls(max_length=config['max_length'])


class PhoBertDataset(Dataset):
    """
    PyTorch Dataset for PhoBERT fine-tuning.
    """
    
    def __init__(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: np.ndarray
    ):
        """
        Initialize dataset.
        
        Args:
            input_ids: Tokenized input IDs tensor
            attention_mask: Attention mask tensor
            labels: Array of labels
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


def extract_phobert_features(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    max_length: int = None
) -> dict:
    """
    Extract PhoBERT features from train/val/test datasets.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        output_dir: Directory to save features
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized features and labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    log.info("Loading datasets...")
    train_df = load_csv(train_path, required_columns=['text', 'label'])
    val_df = load_csv(val_path, required_columns=['text', 'label'])
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    log.info(f"Train: {len(train_df)} samples")
    log.info(f"Val: {len(val_df)} samples")
    log.info(f"Test: {len(test_df)} samples")
    
    # Initialize tokenizer (use cfg.PHOBERT.max_seq_len when max_length is None)
    extractor = PhoBertFeatureExtractor(max_length=max_length)
    
    # Tokenize texts
    log.info("\nTokenizing texts...")
    train_encoded = extractor.tokenize(train_df['text'])
    val_encoded = extractor.tokenize(val_df['text'])
    test_encoded = extractor.tokenize(test_df['text'])
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Save config
    config_path = os.path.join(output_dir, 'phobert_config.pkl')
    extractor.save_config(config_path)
    
    # Save features
    features_path = os.path.join(output_dir, 'phobert_features.pkl')
    joblib.dump({
        'train_input_ids': train_encoded['input_ids'],
        'train_attention_mask': train_encoded['attention_mask'],
        'val_input_ids': val_encoded['input_ids'],
        'val_attention_mask': val_encoded['attention_mask'],
        'test_input_ids': test_encoded['input_ids'],
        'test_attention_mask': test_encoded['attention_mask'],
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }, features_path)
    log.info(f"Saved features to {features_path}")
    
    return {
        'train_encoded': train_encoded,
        'val_encoded': val_encoded,
        'test_encoded': test_encoded,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'extractor': extractor
    }


def create_phobert_data_loaders(
    train_encoded: Dict[str, torch.Tensor],
    val_encoded: Dict[str, torch.Tensor],
    test_encoded: Dict[str, torch.Tensor],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for PhoBERT training.
    
    Args:
        train_encoded, val_encoded, test_encoded: Encoded features
        y_train, y_val, y_test: Labels
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """


    train_dataset = PhoBertDataset(
        train_encoded['input_ids'],
        train_encoded['attention_mask'],
        y_train
    )
    val_dataset = PhoBertDataset(
        val_encoded['input_ids'],
        val_encoded['attention_mask'],
        y_val
    )
    test_dataset = PhoBertDataset(
        test_encoded['input_ids'],
        test_encoded['attention_mask'],
        y_test
    )
    
    # Default to config batch size when not provided
    bs = int(batch_size or cfg.PHOBERT.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    features = extract_phobert_features(
        train_path=os.path.join(cfg.PATHS.splits_dir, 'train.csv'),
        val_path=os.path.join(cfg.PATHS.splits_dir, 'val.csv'),
        test_path=os.path.join(cfg.PATHS.splits_dir, 'test.csv'),
        output_dir=cfg.PATHS.phobert_dir,
        max_length=cfg.PHOBERT.max_seq_len
    )
    
    print("\n" + "="*50)
    print("PhoBERT Feature Extraction Complete!")
    print("="*50)
    print(f"Train input shape: {features['train_encoded']['input_ids'].shape}")
    print(f"Val input shape: {features['val_encoded']['input_ids'].shape}")
    print(f"Test input shape: {features['test_encoded']['input_ids'].shape}")
