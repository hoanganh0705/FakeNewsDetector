"""
PhoBERT Feature Extraction for Transformer Model

PhoBERT is a pre-trained language model for Vietnamese, based on RoBERTa architecture.
This module provides tokenization and dataset creation for PhoBERT fine-tuning.
"""

import pandas as pd
import numpy as np
import os
import pickle
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class PhoBertFeatureExtractor:
    """
    PhoBERT tokenizer wrapper for Vietnamese text classification.
    
    Uses vinai/phobert-base tokenizer to convert text to input IDs and attention masks.
    """
    
    MODEL_NAME = "vinai/phobert-base"
    
    def __init__(self, max_length: int = 256):
        """
        Initialize PhoBERT tokenizer.
        
        Args:
            max_length: Maximum sequence length (PhoBERT max is 256)
        """
        self.max_length = min(max_length, 256)  # PhoBERT max is 256
        
        print(f"Loading PhoBERT tokenizer ({self.MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        print(f"✅ Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")
    
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
        print(f"Tokenizing {len(texts)} documents...")
        
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
        
        print(f"✅ Tokenized to shape: {encoded['input_ids'].shape}")
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def save_config(self, path: str) -> None:
        """Save configuration (tokenizer is loaded from HuggingFace)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_name': self.MODEL_NAME,
                'max_length': self.max_length
            }, f)
        print(f"✅ Saved config to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PhoBertFeatureExtractor':
        """Load extractor from config."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
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
    max_length: int = 256
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
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Initialize tokenizer
    extractor = PhoBertFeatureExtractor(max_length=max_length)
    
    # Tokenize texts
    print("\nTokenizing texts...")
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
    with open(features_path, 'wb') as f:
        pickle.dump({
            'train_input_ids': train_encoded['input_ids'],
            'train_attention_mask': train_encoded['attention_mask'],
            'val_input_ids': val_encoded['input_ids'],
            'val_attention_mask': val_encoded['attention_mask'],
            'test_input_ids': test_encoded['input_ids'],
            'test_attention_mask': test_encoded['attention_mask'],
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }, f)
    print(f"✅ Saved features to {features_path}")
    
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
    batch_size: int = 16
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    features = extract_phobert_features(
        train_path=os.path.join(BASE_DIR, 'data', 'splits', 'train.csv'),
        val_path=os.path.join(BASE_DIR, 'data', 'splits', 'val.csv'),
        test_path=os.path.join(BASE_DIR, 'data', 'splits', 'test.csv'),
        output_dir=os.path.join(BASE_DIR, 'data', 'features', 'phobert'),
        max_length=256
    )
    
    print("\n" + "="*50)
    print("PhoBERT Feature Extraction Complete!")
    print("="*50)
    print(f"Train input shape: {features['train_encoded']['input_ids'].shape}")
    print(f"Val input shape: {features['val_encoded']['input_ids'].shape}")
    print(f"Test input shape: {features['test_encoded']['input_ids'].shape}")
