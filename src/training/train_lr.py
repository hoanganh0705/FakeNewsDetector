"""
Logistic Regression Training Script for Vietnamese Fake News Detection

This script trains a Logistic Regression classifier using TF-IDF features.
Includes hyperparameter tuning with cross-validation.
"""

import os
import sys
import pickle
import json
import time
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.evaluation.metrics import compute_metrics, save_metrics, print_metrics


class LogisticRegressionTrainer:
    """Trainer for Logistic Regression model."""
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the trainer.
        
        Args:
            C: Regularization strength (inverse)
            max_iter: Maximum iterations for solver
            class_weight: 'balanced' to handle class imbalance
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.best_params = None
        self.training_history = {}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'LogisticRegressionTrainer':
        """
        Train the Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            self
        """
        print("Training Logistic Regression...")
        start_time = time.time()
        
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            solver='lbfgs'
        )
        
        self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_prob = self.model.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
        
        self.training_history['train_time'] = train_time
        self.training_history['train_metrics'] = train_metrics
        
        print(f"✅ Training complete in {train_time:.2f}s")
        print(f"   Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"   Train F1: {train_metrics['f1_macro']:.4f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_prob = self.model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
            self.training_history['val_metrics'] = val_metrics
            
            print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"   Val F1: {val_metrics['f1_macro']:.4f}")
        
        return self
    
    def train_with_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict = None,
        cv: int = 5
    ) -> 'LogisticRegressionTrainer':
        """
        Train with hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            self
        """
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000]
            }
        
        print("Running GridSearchCV for Logistic Regression...")
        print(f"Parameter grid: {param_grid}")
        
        start_time = time.time()
        
        base_model = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_splitter,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        search_time = time.time() - start_time
        
        print(f"\n✅ GridSearchCV complete in {search_time:.2f}s")
        print(f"   Best params: {self.best_params}")
        print(f"   Best CV F1: {grid_search.best_score_:.4f}")
        
        self.training_history['grid_search_time'] = search_time
        self.training_history['best_params'] = self.best_params
        self.training_history['best_cv_score'] = grid_search.best_score_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate on a dataset."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return compute_metrics(y, y_pred, y_prob)
    
    def save(self, path: str) -> None:
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'best_params': self.best_params,
                'training_history': self.training_history
            }, f)
        print(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LogisticRegressionTrainer':
        """Load a saved model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls()
        trainer.model = data['model']
        trainer.best_params = data['best_params']
        trainer.training_history = data['training_history']
        return trainer


def main():
    """Main training function."""
    
    print("="*60)
    print("🔬 LOGISTIC REGRESSION TRAINING")
    print("="*60)
    
    # Paths
    features_path = os.path.join(BASE_DIR, 'data', 'features', 'tfidf', 'tfidf_features.pkl')
    model_dir = os.path.join(BASE_DIR, 'experiments', 'lr')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load features
    print("\nLoading TF-IDF features...")
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    X_train = features['X_train']
    X_val = features['X_val']
    X_test = features['X_test']
    y_train = features['y_train']
    y_val = features['y_val']
    y_test = features['y_test']
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Initialize trainer
    trainer = LogisticRegressionTrainer(
        class_weight='balanced',
        random_state=42
    )
    
    # Train with GridSearchCV
    print("\n" + "-"*60)
    trainer.train_with_grid_search(
        X_train, y_train,
        param_grid={
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        },
        cv=5
    )
    
    # Evaluate on validation set
    print("\n" + "-"*60)
    print("📊 Validation Results:")
    val_metrics = trainer.evaluate(X_val, y_val)
    print_metrics(val_metrics)
    
    # Evaluate on test set
    print("\n" + "-"*60)
    print("📊 Test Results:")
    test_metrics = trainer.evaluate(X_test, y_test)
    print_metrics(test_metrics)
    
    # Save model
    model_path = os.path.join(model_dir, 'lr_model.pkl')
    trainer.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    save_metrics({
        'model': 'Logistic Regression',
        'best_params': trainer.best_params,
        'validation': val_metrics,
        'test': test_metrics,
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
