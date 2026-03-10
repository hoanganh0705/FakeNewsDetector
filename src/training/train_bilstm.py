"""
BiLSTM Training Script for Vietnamese Fake News Detection


This script trains a Bidirectional LSTM model using word embeddings.
Includes early stopping and learning rate scheduling.
"""

import os
import joblib
import time
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.features.embedding_features import TextDataset, collate_fn
from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.bilstm_model import BiLSTMClassifier
from src.training.runner import save_training_results
from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


# BiLSTMClassifier is defined in src/models/bilstm_model.py
# and imported above — keeping training logic and architecture separate.


class BiLSTMTrainer:
    """Trainer for BiLSTM model."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
        learning_rate: float = None,
        weight_decay: float = None,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension (defaults to cfg.BILSTM.embedding_dim)
            hidden_dim: LSTM hidden dimension (defaults to cfg.BILSTM.hidden_dim)
            num_layers: Number of LSTM layers (defaults to cfg.BILSTM.num_layers)
            dropout: Dropout rate (defaults to cfg.BILSTM.dropout)
            learning_rate: Learning rate (defaults to cfg.BILSTM.learning_rate)
            weight_decay: L2 regularization (defaults to cfg.BILSTM.weight_decay)
            device: Device to use ('cuda' or 'cpu')
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim if embedding_dim is not None else cfg.BILSTM.embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else cfg.BILSTM.hidden_dim
        self.num_layers = num_layers if num_layers is not None else cfg.BILSTM.num_layers
        self.dropout = dropout if dropout is not None else cfg.BILSTM.dropout
        self.learning_rate = learning_rate if learning_rate is not None else cfg.BILSTM.learning_rate
        self.weight_decay = weight_decay if weight_decay is not None else cfg.BILSTM.weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        log.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        self.best_val_f1 = 0
        self.best_model_state = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        patience: int = None,
        class_weights: np.ndarray = None
    ) -> 'BiLSTMTrainer':
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            class_weights: Class weights for imbalanced data
            
        Returns:
            self
        """
        epochs = epochs if epochs is not None else cfg.BILSTM.epochs
        patience = patience if patience is not None else cfg.BILSTM.patience

        # Setup loss function with class weights and label smoothing
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.BILSTM.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.BILSTM.label_smoothing)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Use Cosine Annealing schedule instead of ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        log.info(f"\nTraining BiLSTM for {epochs} epochs...")
        log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        best_val_f1 = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (sequences, attention_mask, labels) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                try:
                    outputs = self.model(sequences, attention_mask)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log.error(
                        "CUDA out of memory during forward/backward pass! "
                        "Try reducing batch_size (current: %d) or "
                        "max_seq_length in config.py.",
                        train_loader.batch_size,
                    )
                    raise

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - epoch_start
            
            log.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_val_f1 = val_f1
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                log.info(f"    New best model! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.warning(f"\n Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        log.info(f"\n Training complete in {total_time:.2f}s")
        log.info(f"Best Val F1: {self.best_val_f1:.4f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.training_history['total_time'] = total_time
        
        return self
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, attention_mask, labels in data_loader:
                sequences = sequences.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, metrics['accuracy'], metrics['f1_macro']
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and probabilities."""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for sequences, attention_mask, labels in data_loader:
                sequences = sequences.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(sequences, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def evaluate(self, data_loader: DataLoader, y_true: np.ndarray) -> dict:
        """Evaluate on a dataset."""
        y_pred, y_prob = self.predict(data_loader)
        return compute_metrics(y_true, y_pred, y_prob)
    
    def save(self, path: str) -> None:
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'training_history': self.training_history,
            'best_val_f1': self.best_val_f1
        }, path)
        log.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'BiLSTMTrainer':
        """Load a saved model."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        trainer = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            device=device
        )
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.training_history = checkpoint['training_history']
        trainer.best_val_f1 = checkpoint['best_val_f1']
        
        return trainer


def main():
    """Main training function."""
    from src.utils.common import set_reproducibility_seeds

    set_reproducibility_seeds()

    log.info("=" * 60)
    log.info("BiLSTM TRAINING")
    log.info("=" * 60)

    # Paths
    features_path = os.path.join(cfg.PATHS.embedding_dir, 'embedding_features.pkl')
    extractor_path = os.path.join(cfg.PATHS.embedding_dir, 'embedding_extractor.pkl')  # also used in FastText block
    model_dir = cfg.PATHS.bilstm_dir
    os.makedirs(model_dir, exist_ok=True)

    # Load features
    log.info("Loading embedding features...")
    features = joblib.load(features_path)

    train_sequences = features['train_sequences']
    val_sequences = features['val_sequences']
    test_sequences = features['test_sequences']
    y_train = features['y_train']
    y_val = features['y_val']
    y_test = features['y_test']
    vocab_size = features['vocab_size']

    log.info("Vocabulary size: %d", vocab_size)
    log.info("Train samples: %d", len(train_sequences))
    log.info("Val samples: %d", len(val_sequences))
    log.info("Test samples: %d", len(test_sequences))

    # Create data loaders
    batch_size = cfg.BILSTM.batch_size

    train_dataset = TextDataset(train_sequences, y_train)
    val_dataset   = TextDataset(val_sequences,   y_val)
    test_dataset  = TextDataset(test_sequences,  y_test)

    _num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=_num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=_num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=_num_workers, pin_memory=True)

    # Compute class weights based on config
    class_weights = None
    if cfg.BILSTM.class_weight == 'balanced':
        from src.utils.common import compute_balanced_class_weights
        class_weights = compute_balanced_class_weights(y_train)
    log.info("Class weights: %s", class_weights)

    # Initialize trainer (defaults pulled from cfg.BILSTM)
    trainer = BiLSTMTrainer(vocab_size=vocab_size)

    # If FastText path is provided, load matrix and set embeddings (requires extractor)
    if cfg.BILSTM.fasttext_path:
        try:
            if os.path.exists(extractor_path):
                from src.features.embedding_features import EmbeddingFeatureExtractor, load_fasttext_matrix
                log.info("Loading FastText embeddings from %s ...", cfg.BILSTM.fasttext_path)
                ext = EmbeddingFeatureExtractor.load(extractor_path)
                matrix = load_fasttext_matrix(ext.vocab, cfg.BILSTM.fasttext_path, cfg.BILSTM.embedding_dim)
                trainer.model.load_pretrained_embeddings(matrix)
                log.info("FastText embeddings loaded into model")
                # Optionally freeze embeddings to prevent overfitting
                if getattr(cfg.BILSTM, 'freeze_embeddings', False):
                    trainer.model.embedding.weight.requires_grad = False
                    log.info("Embedding layer frozen (freeze_embeddings=True)")
            else:
                log.info("FastText path set but embedding extractor not found at %s; skipping pretrained init", extractor_path)
        except (ImportError, FileNotFoundError, RuntimeError, OSError) as e:
            log.warning("Could not load FastText embeddings: %s", e)

    # Train
    log.info("-" * 60)
    trainer.train(
        train_loader,
        val_loader,
        class_weights=class_weights
    )

    # Evaluate on validation set
    log.info("-" * 60)
    log.info("Validation Results:")
    val_metrics = trainer.evaluate(val_loader, y_val)
    print_metrics(val_metrics)

    # Evaluate on test set — compute predictions once, reuse for metrics and saving
    log.info("-" * 60)
    log.info("Test Results:")
    y_pred, y_prob = trainer.predict(test_loader)
    test_metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(test_metrics)

    # Save model
    model_path = os.path.join(model_dir, 'bilstm_model.pt')
    trainer.save(model_path)

    # Save results (metrics, predictions, experiment log)
    save_training_results(
        model_name='BiLSTM',
        model_dir=model_dir,
        model_path=model_path,
        metrics_dict={
            'model': 'BiLSTM',
            'config': {
                'vocab_size':     vocab_size,
                'embedding_dim':  trainer.embedding_dim,
                'hidden_dim':     trainer.hidden_dim,
                'num_layers':     trainer.num_layers,
                'dropout':        trainer.dropout,
                'batch_size':     cfg.BILSTM.batch_size,
                'learning_rate':  trainer.learning_rate,
                'epochs':         cfg.BILSTM.epochs,
                'patience':       cfg.BILSTM.patience,
            },
            'validation': val_metrics,
            'test': test_metrics,
            'training_history': {
                'best_val_f1':    trainer.best_val_f1,
                'epochs_trained': len(trainer.training_history['train_loss']),
            },
        },
        test_metrics=test_metrics,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        experiment_config={
            'embedding_dim': trainer.embedding_dim,
            'hidden_dim':    trainer.hidden_dim,
            'num_layers':    trainer.num_layers,
            'dropout':       trainer.dropout,
            'learning_rate': trainer.learning_rate,
            'batch_size':    cfg.BILSTM.batch_size,
            'epochs':        cfg.BILSTM.epochs,
        },
    )

    return trainer, test_metrics


if __name__ == "__main__":
    trainer, metrics = main()
