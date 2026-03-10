"""
Word Embedding Feature Extraction for BiLSTM Model


This module provides word embeddings (Word2Vec/FastText) for sequence models.
Converts text into sequences of word vectors for BiLSTM training.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Optional, List
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import cfg
from src.utils.common import load_csv

from src.utils.logger import get_logger
log = get_logger(__name__)


class Vocabulary:
    """
    Vocabulary class for mapping words to indices and vice versa.
    """
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    PAD_IDX = 0
    UNK_IDX = 1
    
    def __init__(self, max_size: int = 50000, min_freq: int = 2):
        """
        Initialize vocabulary.
        
        Args:
            max_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include
        """
        self.max_size = max_size
        self.min_freq = min_freq
        
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.word_freq = Counter()
        
    def build(self, texts: pd.Series) -> 'Vocabulary':
        """
        Build vocabulary from texts.
        
        Args:
            texts: Series of text documents
            
        Returns:
            self
        """
        # Count word frequencies
        for text in texts:
            words = str(text).split()
            self.word_freq.update(words)
        
        # Filter by frequency and take top max_size words
        valid_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ][:self.max_size - 2]  # -2 for PAD and UNK
        
        # Build word2idx and idx2word
        for idx, word in enumerate(valid_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        log.info(f"Built vocabulary with {len(self.word2idx)} words")
        return self
    
    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to list of word indices."""
        words = str(text).split()
        return [self.word2idx.get(word, self.UNK_IDX) for word in words]
    
    def __len__(self):
        return len(self.word2idx)


class EmbeddingFeatureExtractor:
    """
    Word embedding feature extractor for BiLSTM model.
    
    Converts text to sequences of indices that can be fed to an embedding layer.
    """
    
    def __init__(
        self,
        max_vocab_size: int = 50000,
        max_seq_length: int = 512,
        min_freq: int = 2,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize the embedding feature extractor.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length (truncate longer texts)
            min_freq: Minimum word frequency to include in vocabulary
            embedding_dim: Dimension of word embeddings
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.min_freq = min_freq
        # Use central config embedding dim when not provided
        self.embedding_dim = embedding_dim if embedding_dim is not None else cfg.BILSTM.embedding_dim
        
        self.vocab = Vocabulary(max_size=max_vocab_size, min_freq=min_freq)
        self.is_fitted = False
    
    def fit(self, texts: pd.Series) -> 'EmbeddingFeatureExtractor':
        """
        Build vocabulary from training texts.
        
        Args:
            texts: Series of text documents
            
        Returns:
            self
        """
        log.info(f"Building vocabulary from {len(texts)} documents...")
        self.vocab.build(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: pd.Series) -> List[List[int]]:
        """
        Transform texts to sequences of word indices.
        
        Args:
            texts: Series of text documents
            
        Returns:
            List of sequences (each sequence is a list of word indices)
        """
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        
        sequences = []
        for text in texts:
            indices = self.vocab.text_to_indices(text)
            # Truncate to max_seq_length
            if len(indices) > self.max_seq_length:
                indices = indices[:self.max_seq_length]
            sequences.append(indices)
        
        log.info(f"Transformed {len(texts)} documents to sequences")
        return sequences
    
    def fit_transform(self, texts: pd.Series) -> List[List[int]]:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def save(self, path: str) -> None:
        """Save the extractor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'vocab': self.vocab,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length,
            'min_freq': self.min_freq,
            'embedding_dim': self.embedding_dim,
            'is_fitted': self.is_fitted
        }, path)
        log.info(f"Saved embedding extractor to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EmbeddingFeatureExtractor':
        """Load extractor from disk."""
        data = joblib.load(path)
        
        extractor = cls(
            max_vocab_size=data['max_vocab_size'],
            max_seq_length=data['max_seq_length'],
            min_freq=data['min_freq'],
            embedding_dim=data['embedding_dim']
        )
        extractor.vocab = data['vocab']
        extractor.is_fitted = data['is_fitted']
        
        log.info(f"Loaded embedding extractor from {path}")
        return extractor


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification with BiLSTM.
    """
    
    def __init__(self, sequences: List[List[int]], labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: List of word index sequences
            labels: Array of labels
        """
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def collate_fn(batch):
    """
    Collate function for DataLoader that pads sequences to same length.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        Padded sequences tensor and labels tensor
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (sequences_padded != 0).long()
    
    return sequences_padded, attention_mask, labels


def extract_embedding_features(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    max_vocab_size: int = 50000,
    max_seq_length: Optional[int] = None,
    min_freq: int = 2
) -> dict:
    """
    Extract embedding features from train/val/test datasets.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        output_dir: Directory to save features
        max_vocab_size: Maximum vocabulary size
        max_seq_length: Maximum sequence length
        min_freq: Minimum word frequency
        
    Returns:
        Dictionary with sequences, labels, and extractor
    """
    os.makedirs(output_dir, exist_ok=True)
    max_seq_length = int(max_seq_length or cfg.BILSTM.max_seq_length)
    
    # Load datasets
    log.info("Loading datasets...")
    train_df = load_csv(train_path, required_columns=['text', 'label'])
    val_df = load_csv(val_path, required_columns=['text', 'label'])
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    log.info(f"Train: {len(train_df)} samples")
    log.info(f"Val: {len(val_df)} samples")
    log.info(f"Test: {len(test_df)} samples")
    
    # Initialize and fit extractor
    extractor = EmbeddingFeatureExtractor(
        max_vocab_size=max_vocab_size,
        max_seq_length=max_seq_length,
        min_freq=min_freq
    )
    
    # Extract features
    log.info("\nExtracting embedding features...")
    train_sequences = extractor.fit_transform(train_df['text'])
    val_sequences = extractor.transform(val_df['text'])
    test_sequences = extractor.transform(test_df['text'])
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Save extractor
    extractor_path = os.path.join(output_dir, 'embedding_extractor.pkl')
    extractor.save(extractor_path)
    
    # Save features
    features_path = os.path.join(output_dir, 'embedding_features.pkl')
    joblib.dump({
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'test_sequences': test_sequences,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'vocab_size': extractor.vocab_size
    }, features_path)
    log.info(f"Saved features to {features_path}")
    
    # Calculate sequence length statistics
    train_lengths = [len(seq) for seq in train_sequences]
    log.info(f"\nSequence length statistics (train):")
    log.info(f"Min: {min(train_lengths)}, Max: {max(train_lengths)}, Mean: {np.mean(train_lengths):.0f}")
    
    return {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'test_sequences': test_sequences,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'extractor': extractor
    }


def create_data_loaders(
    train_sequences: List[List[int]],
    val_sequences: List[List[int]],
    test_sequences: List[List[int]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training.
    
    Args:
        train_sequences, val_sequences, test_sequences: Sequences for each split
        y_train, y_val, y_test: Labels for each split
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TextDataset(train_sequences, y_train)
    val_dataset = TextDataset(val_sequences, y_val)
    test_dataset = TextDataset(test_sequences, y_test)
    
    bs = int(batch_size or cfg.BILSTM.batch_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=bs, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=bs, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=bs, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def load_fasttext_matrix(vocab, fasttext_path: str, dim: int) -> np.ndarray:
    """
    Load FastText .bin model and build an embedding matrix aligned with `vocab`.

    Args:
        vocab: Vocabulary instance (with .word2idx mapping) or a mapping {word: idx}
        fasttext_path: Path to the FastText .bin file
        dim: Embedding dimensionality

    Returns:
        numpy array of shape (vocab_size, dim)
    """


    try:
        import fasttext
    except ImportError:
        raise ImportError("fasttext library is required to load FastText .bin files")

    # Determine mapping
    if hasattr(vocab, 'word2idx'):
        mapping = vocab.word2idx
    elif isinstance(vocab, dict):
        mapping = vocab
    else:
        raise ValueError("vocab must have .word2idx or be a dict mapping")

    model = fasttext.load_model(fasttext_path)

    vocab_size = max(mapping.values()) + 1
    matrix = np.random.normal(scale=0.6, size=(vocab_size, dim)).astype(np.float32)

    for word, idx in mapping.items():
        if idx >= vocab_size:
            continue
        try:
            vec = model.get_word_vector(word)
            if vec is not None and len(vec) == dim:
                matrix[idx] = vec
        except (KeyError, ValueError):
            # keep random init for missing words
            continue

    return matrix


if __name__ == "__main__":
    features = extract_embedding_features(
        train_path=os.path.join(cfg.PATHS.splits_dir, 'train.csv'),
        val_path=os.path.join(cfg.PATHS.splits_dir, 'val.csv'),
        test_path=os.path.join(cfg.PATHS.splits_dir, 'test.csv'),
        output_dir=cfg.PATHS.embedding_dir,
        max_vocab_size=50000,
        max_seq_length=cfg.BILSTM.max_seq_length,
        min_freq=2
    )
    
    print("\n" + "="*50)
    print("Embedding Feature Extraction Complete!")
    print("="*50)
    print(f"Vocabulary size: {features['extractor'].vocab_size}")
    print(f"Train sequences: {len(features['train_sequences'])}")
    print(f"Val sequences: {len(features['val_sequences'])}")
    print(f"Test sequences: {len(features['test_sequences'])}")
