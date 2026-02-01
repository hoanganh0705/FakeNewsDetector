"""
BiLSTM Training Script for Vietnamese Fake News Detection

This script trains a Bidirectional LSTM model using word embeddings.
Includes early stopping and learning rate scheduling.
"""

import os
import sys
import pickle
import json
import time
import numpy as np
from datetime import datetime
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.features.embedding_features import TextDataset, collate_fn
from src.evaluation.metrics import compute_metrics, save_metrics, print_metrics


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM model for text classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0
    ):
        """
        Initialize the BiLSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            padding_idx: Index of padding token
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            attention_mask: Mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)
        
        # Attention
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.unsqueeze(-1) == 0, 
                float('-inf')
            )
        
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(context)  # (batch, num_classes)
        
        return output


class BiLSTMTrainer:
    """Trainer for BiLSTM model."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: Device to use ('cuda' or 'cpu')
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
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
        epochs: int = 20,
        patience: int = 5,
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
        # Setup loss function with class weights
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        print(f"\nTraining BiLSTM for {epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
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
                
                outputs = self.model(sequences, attention_mask)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
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
            self.scheduler.step(val_f1)
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"   ✅ New best model! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training complete in {total_time:.2f}s")
        print(f"   Best Val F1: {self.best_val_f1:.4f}")
        
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
        print(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'BiLSTMTrainer':
        """Load a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        
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
    
    print("="*60)
    print("🔬 BiLSTM TRAINING")
    print("="*60)
    
    # Paths
    features_path = os.path.join(BASE_DIR, 'data', 'features', 'embedding', 'embedding_features.pkl')
    extractor_path = os.path.join(BASE_DIR, 'data', 'features', 'embedding', 'embedding_extractor.pkl')
    model_dir = os.path.join(BASE_DIR, 'experiments', 'bilstm')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load features
    print("\nLoading embedding features...")
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    train_sequences = features['train_sequences']
    val_sequences = features['val_sequences']
    test_sequences = features['test_sequences']
    y_train = features['y_train']
    y_val = features['y_val']
    y_test = features['y_test']
    vocab_size = features['vocab_size']
    
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Train samples: {len(train_sequences)}")
    print(f"  Val samples: {len(val_sequences)}")
    print(f"  Test samples: {len(test_sequences)}")
    
    # Create data loaders
    batch_size = 32
    
    train_dataset = TextDataset(train_sequences, y_train)
    val_dataset = TextDataset(val_sequences, y_val)
    test_dataset = TextDataset(test_sequences, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"  Class weights: {class_weights}")
    
    # Initialize trainer
    trainer = BiLSTMTrainer(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=1e-3
    )
    
    # Train
    print("\n" + "-"*60)
    trainer.train(
        train_loader,
        val_loader,
        epochs=20,
        patience=5,
        class_weights=class_weights
    )
    
    # Evaluate on validation set
    print("\n" + "-"*60)
    print("📊 Validation Results:")
    val_metrics = trainer.evaluate(val_loader, y_val)
    print_metrics(val_metrics)
    
    # Evaluate on test set
    print("\n" + "-"*60)
    print("📊 Test Results:")
    test_metrics = trainer.evaluate(test_loader, y_test)
    print_metrics(test_metrics)
    
    # Save model
    model_path = os.path.join(model_dir, 'bilstm_model.pt')
    trainer.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    save_metrics({
        'model': 'BiLSTM',
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3
        },
        'validation': val_metrics,
        'test': test_metrics,
        'training_history': {
            'best_val_f1': trainer.best_val_f1,
            'epochs_trained': len(trainer.training_history['train_loss'])
        },
        'timestamp': datetime.now().isoformat()
    }, metrics_path)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    return trainer, test_metrics


if __name__ == "__main__":
    trainer, metrics = main()
