"""
PhoBERT Training Script for Vietnamese Fake News Detection

This script fine-tunes PhoBERT (Pre-trained language model for Vietnamese)
for fake news classification task.
"""

import os
import sys
import pickle
import json
import time
import numpy as np
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.features.phobert_features import PhoBertDataset
from src.evaluation.metrics import compute_metrics, save_metrics, print_metrics


class PhoBertClassifier(nn.Module):
    """
    PhoBERT-based classifier for Vietnamese fake news detection.
    """
    
    MODEL_NAME = "vinai/phobert-base"
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Initialize the PhoBERT classifier.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
        """
        super(PhoBertClassifier, self).__init__()
        
        print(f"Loading PhoBERT model ({self.MODEL_NAME})...")
        self.bert = AutoModel.from_pretrained(self.MODEL_NAME)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("   BERT parameters frozen")
        
        hidden_size = self.bert.config.hidden_size  # 768 for phobert-base
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        print(f"✅ Model initialized. Hidden size: {hidden_size}")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        
        # Classification
        logits = self.classifier(cls_output)  # (batch, num_classes)
        
        return logits


class PhoBertTrainer:
    """Trainer for PhoBERT model."""
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        freeze_bert: bool = False,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
            learning_rate: Learning rate (typically 2e-5 for BERT fine-tuning)
            weight_decay: L2 regularization
            warmup_ratio: Ratio of warmup steps
            freeze_bert: Whether to freeze BERT layers
            device: Device to use
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = PhoBertClassifier(
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=freeze_bert
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
        epochs: int = 5,
        patience: int = 3,
        class_weights: np.ndarray = None,
        gradient_accumulation_steps: int = 1
    ) -> 'PhoBertTrainer':
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            class_weights: Class weights for imbalanced data
            gradient_accumulation_steps: Steps to accumulate gradients
            
        Returns:
            self
        """
        # Setup loss function
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Setup scheduler with warmup
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\nTraining PhoBERT for {epochs} epochs...")
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
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
            
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * gradient_accumulation_steps
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Progress update
                if (batch_idx + 1) % 50 == 0:
                    print(f"   Batch {batch_idx + 1}/{len(train_loader)}, "
                          f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"   Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_val_f1 = val_f1
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
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
            self.model.to(self.device)
        
        self.training_history['total_time'] = total_time
        
        return self
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
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
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
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
            'training_history': self.training_history,
            'best_val_f1': self.best_val_f1
        }, path)
        print(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'PhoBertTrainer':
        """Load a saved model."""
        checkpoint = torch.load(path, map_location='cpu')
        
        trainer = cls(device=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.training_history = checkpoint['training_history']
        trainer.best_val_f1 = checkpoint['best_val_f1']
        
        return trainer


def main():
    """Main training function."""
    
    print("="*60)
    print("🔬 PhoBERT TRAINING")
    print("="*60)
    
    # Paths
    features_path = os.path.join(BASE_DIR, 'data', 'features', 'phobert', 'phobert_features.pkl')
    model_dir = os.path.join(BASE_DIR, 'experiments', 'bert')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load features
    print("\nLoading PhoBERT features...")
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    y_train = features['y_train']
    y_val = features['y_val']
    y_test = features['y_test']
    
    print(f"  Train samples: {len(y_train)}")
    print(f"  Val samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    
    # Create datasets
    train_dataset = PhoBertDataset(
        features['train_input_ids'],
        features['train_attention_mask'],
        y_train
    )
    val_dataset = PhoBertDataset(
        features['val_input_ids'],
        features['val_attention_mask'],
        y_val
    )
    test_dataset = PhoBertDataset(
        features['test_input_ids'],
        features['test_attention_mask'],
        y_test
    )
    
    # Create data loaders
    batch_size = 16  # Smaller batch for BERT due to memory
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"  Class weights: {class_weights}")
    
    # Initialize trainer
    trainer = PhoBertTrainer(
        num_classes=2,
        dropout=0.1,
        learning_rate=2e-5,
        warmup_ratio=0.1
    )
    
    # Train
    print("\n" + "-"*60)
    trainer.train(
        train_loader,
        val_loader,
        epochs=5,
        patience=3,
        class_weights=class_weights,
        gradient_accumulation_steps=2
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
    model_path = os.path.join(model_dir, 'phobert_model.pt')
    trainer.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    save_metrics({
        'model': 'PhoBERT',
        'config': {
            'model_name': 'vinai/phobert-base',
            'dropout': 0.1,
            'learning_rate': 2e-5,
            'batch_size': batch_size
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
