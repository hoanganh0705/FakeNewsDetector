"""
PhoBERT Training Script for Vietnamese Fake News Detection


This script fine-tunes PhoBERT (Pre-trained language model for Vietnamese)
for fake news classification task.
"""

import os
import re
import joblib
import time
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.features.phobert_features import PhoBertDataset
from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.phobert_model import PhoBertClassifier
from src.training.runner import save_training_results
from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


# PhoBertClassifier is defined in src/models/phobert_model.py
# and imported above — keeping training logic and architecture separate.


class PhoBertTrainer:
    """Trainer for PhoBERT model."""
    
    def __init__(
        self,
        num_classes: int = None,
        dropout: float = None,
        learning_rate: float = None,
        weight_decay: float = None,
        warmup_ratio: float = None,
        freeze_bert: bool = False,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            num_classes: Number of output classes (defaults to cfg.PHOBERT.num_classes)
            dropout: Dropout rate (defaults to cfg.PHOBERT.dropout)
            learning_rate: Learning rate (defaults to cfg.PHOBERT.learning_rate)
            weight_decay: L2 regularization (defaults to cfg.PHOBERT.weight_decay)
            warmup_ratio: Ratio of warmup steps (defaults to cfg.PHOBERT.warmup_ratio)
            freeze_bert: Whether to freeze BERT layers
            device: Device to use
        """
        self.learning_rate = learning_rate if learning_rate is not None else cfg.PHOBERT.learning_rate
        self.weight_decay = weight_decay if weight_decay is not None else cfg.PHOBERT.weight_decay
        self.warmup_ratio = warmup_ratio if warmup_ratio is not None else cfg.PHOBERT.warmup_ratio
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        log.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = PhoBertClassifier(
            num_classes=num_classes if num_classes is not None else cfg.PHOBERT.num_classes,
            dropout=dropout if dropout is not None else cfg.PHOBERT.dropout,
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
        # Saved state dicts for resuming training (populated by load())
        self._saved_optimizer_state = None
        self._saved_scheduler_state = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        patience: int = None,
        class_weights: np.ndarray = None,
        gradient_accumulation_steps: int = None
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
        epochs = epochs if epochs is not None else cfg.PHOBERT.epochs
        patience = patience if patience is not None else cfg.PHOBERT.patience
        gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else cfg.PHOBERT.gradient_accumulation_steps

        # Setup loss function (with label smoothing)
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.PHOBERT.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.PHOBERT.label_smoothing)

        # Setup optimizer with layer-wise learning rate decay for BERT encoder
        no_decay = ['bias', 'LayerNorm.weight']

        # Collect parameters by encoder layer index
        layer_map = {}
        for n, p in self.model.named_parameters():
            m = re.search(r'encoder.layer.(\d+)', n)
            if m:
                idx = int(m.group(1))
                layer_map.setdefault(idx, []).append((n, p))

        optimizer_grouped_parameters = []
        if len(layer_map) > 0:
            num_layers = max(layer_map.keys()) + 1
            for layer_idx in range(num_layers):
                lr = self.learning_rate * (cfg.PHOBERT.layer_lr_decay ** (num_layers - 1 - layer_idx))
                params_decay = [p for n, p in layer_map.get(layer_idx, []) if not any(nd in n for nd in no_decay)]
                params_no_decay = [p for n, p in layer_map.get(layer_idx, []) if any(nd in n for nd in no_decay)]
                if params_decay:
                    optimizer_grouped_parameters.append({'params': params_decay, 'weight_decay': self.weight_decay, 'lr': lr})
                if params_no_decay:
                    optimizer_grouped_parameters.append({'params': params_no_decay, 'weight_decay': 0.0, 'lr': lr})

        # Embeddings and pooler (slightly lower LR)
        embed_params = [p for n, p in self.model.named_parameters() if 'embeddings' in n]
        pooler_params = [p for n, p in self.model.named_parameters() if 'pooler' in n]
        if embed_params:
            optimizer_grouped_parameters.append({'params': embed_params, 'weight_decay': self.weight_decay, 'lr': self.learning_rate * cfg.PHOBERT.layer_lr_decay})
        if pooler_params:
            optimizer_grouped_parameters.append({'params': pooler_params, 'weight_decay': 0.0, 'lr': self.learning_rate * cfg.PHOBERT.layer_lr_decay})

        # Classifier heads (use base LR)
        classifier_params = [p for n, p in self.model.named_parameters() if n.startswith('classifier') or 'classifier' in n]
        if classifier_params:
            optimizer_grouped_parameters.append({'params': classifier_params, 'weight_decay': self.weight_decay, 'lr': self.learning_rate})

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Setup scheduler with warmup
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Restore optimizer/scheduler state if resuming from a checkpoint
        if self._saved_optimizer_state is not None:
            self.optimizer.load_state_dict(self._saved_optimizer_state)
            self._saved_optimizer_state = None
        if self._saved_scheduler_state is not None:
            self.scheduler.load_state_dict(self._saved_scheduler_state)
            self._saved_scheduler_state = None
        
        log.info(f"\nTraining PhoBERT for {epochs} epochs...")
        log.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
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
            
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                try:
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                    
                    # Gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log.error(
                        "CUDA out of memory during forward/backward pass! "
                        "Try reducing batch_size (current: %d) or "
                        "max_seq_len (current: %d) in config.py.",
                        train_loader.batch_size,
                        cfg.PHOBERT.max_seq_len,
                    )
                    raise
                
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
                    log.info(f"Batch {batch_idx + 1}/{len(train_loader)}, "
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
            
            log.info(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            log.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            log.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
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
        """Save the model, optimizer, and scheduler state for full resumption."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict':     self.model.state_dict(),
            'training_history':     self.training_history,
            'best_val_f1':          self.best_val_f1,
        }
        # Persist optimizer & scheduler so training can resume without LR jump
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        log.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> 'PhoBertTrainer':
        """
        Load a saved model.

        Restores model weights, optimizer state (if present), and scheduler
        state (if present) so training can resume exactly where it stopped.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)

        trainer = cls(device=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.training_history = checkpoint['training_history']
        trainer.best_val_f1      = checkpoint['best_val_f1']

        # Stash optimizer/scheduler state for deferred restore in train()
        # (optimizer & scheduler are None until train() creates them)
        if 'optimizer_state_dict' in checkpoint:
            trainer._saved_optimizer_state = checkpoint['optimizer_state_dict']
        if 'scheduler_state_dict' in checkpoint:
            trainer._saved_scheduler_state = checkpoint['scheduler_state_dict']

        return trainer


def main():
    """Main training function."""
    from src.utils.common import set_reproducibility_seeds

    set_reproducibility_seeds()

    log.info("=" * 60)
    log.info("PhoBERT TRAINING")
    log.info("=" * 60)

    # Paths
    features_path = os.path.join(cfg.PATHS.phobert_dir, 'phobert_features.pkl')
    model_dir = cfg.PATHS.bert_dir
    os.makedirs(model_dir, exist_ok=True)

    # Load features
    log.info("Loading PhoBERT features...")
    features = joblib.load(features_path)

    y_train = features['y_train']
    y_val = features['y_val']
    y_test = features['y_test']

    log.info("Train samples: %d", len(y_train))
    log.info("Val samples: %d", len(y_val))
    log.info("Test samples: %d", len(y_test))

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
    batch_size = cfg.PHOBERT.batch_size  # Smaller batch for BERT due to memory

    _num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=_num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=_num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=_num_workers, pin_memory=True)

    # Compute class weights
    from src.utils.common import compute_balanced_class_weights
    class_weights = compute_balanced_class_weights(y_train)
    log.info("Class weights: %s", class_weights)

    # Initialize trainer (defaults pulled from cfg.PHOBERT)
    trainer = PhoBertTrainer()

    # Train
    log.info("-" * 60)
    trainer.train(
        train_loader,
        val_loader,
        class_weights=class_weights,
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
    model_path = os.path.join(model_dir, 'phobert_model.pt')
    trainer.save(model_path)

    # Save results (metrics, predictions, experiment log)
    save_training_results(
        model_name='PhoBERT',
        model_dir=model_dir,
        model_path=model_path,
        metrics_dict={
            'model': 'PhoBERT',
            'config': {
                'model_name': cfg.PHOBERT.model_name,
                'dropout': cfg.PHOBERT.dropout,
                'learning_rate': cfg.PHOBERT.learning_rate,
                'batch_size': batch_size,
            },
            'validation': val_metrics,
            'test': test_metrics,
            'training_history': {
                'best_val_f1': trainer.best_val_f1,
                'epochs_trained': len(trainer.training_history['train_loss']),
            },
        },
        test_metrics=test_metrics,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        experiment_config={
            'model_name':    cfg.PHOBERT.model_name,
            'dropout':       cfg.PHOBERT.dropout,
            'learning_rate': cfg.PHOBERT.learning_rate,
            'batch_size':    batch_size,
            'epochs':        cfg.PHOBERT.epochs,
            'max_seq_len':   cfg.PHOBERT.max_seq_len,
        },
    )

    return trainer, test_metrics


if __name__ == "__main__":
    trainer, metrics = main()
