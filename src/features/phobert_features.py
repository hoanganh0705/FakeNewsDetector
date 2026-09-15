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
    MODEL_NAME = cfg.PHOBERT.model_name

    def __init__(self, max_length: int = None):
        self.max_length = int(max_length or cfg.PHOBERT.max_seq_len)

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
        log.info(f"Tokenizing {len(texts)} documents...")
        
        text_list = texts.astype(str).tolist()
        
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model_name': self.MODEL_NAME,
            'max_length': self.max_length
        }, path)
        log.info(f"Saved config to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PhoBertFeatureExtractor':
        config = joblib.load(path)
        return cls(max_length=config['max_length'])


class PhoBertDataset(Dataset):
    def __init__(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: np.ndarray
    ):
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
    os.makedirs(output_dir, exist_ok=True)
    
    log.info("Loading datasets...")
    train_df = load_csv(train_path, required_columns=['text', 'label'])
    val_df = load_csv(val_path, required_columns=['text', 'label'])
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    log.info(f"Train: {len(train_df)} samples")
    log.info(f"Val: {len(val_df)} samples")
    log.info(f"Test: {len(test_df)} samples")
    
    extractor = PhoBertFeatureExtractor(max_length=max_length)
    
    log.info("\nTokenizing texts...")
    train_encoded = extractor.tokenize(train_df['text'])
    val_encoded = extractor.tokenize(val_df['text'])
    test_encoded = extractor.tokenize(test_df['text'])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    config_path = os.path.join(output_dir, 'phobert_config.pkl')
    extractor.save_config(config_path)
    
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
