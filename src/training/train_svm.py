"""
SVM Training Script for Vietnamese Fake News Detection


This script trains a Support Vector Machine classifier using TF-IDF features.
Includes hyperparameter tuning with cross-validation.
"""

import os
import joblib
import time
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.evaluation.metrics import compute_metrics, print_metrics
from src.training.runner import save_training_results
from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


class SVMTrainer:
    """Trainer for SVM model."""
    
    def __init__(
        self,
        kernel: str = None,
        C: float = None,
        gamma: str = None,
        class_weight: str = None,
        random_state: int = None,
        use_linear: bool = None
    ):
        """
        Initialize the trainer.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly') (defaults to cfg.SVM.kernel)
            C: Regularization parameter (defaults to cfg.SVM.C)
            gamma: Kernel coefficient (defaults to cfg.SVM.gamma)
            class_weight: 'balanced' to handle class imbalance (defaults to cfg.SVM.class_weight)
            random_state: Random seed (defaults to cfg.RANDOM_STATE)
            use_linear: Use LinearSVC + calibration (defaults to cfg.SVM.use_linear)
        """
        self.kernel = kernel if kernel is not None else cfg.SVM.kernel
        self.C = C if C is not None else cfg.SVM.C
        self.gamma = gamma if gamma is not None else cfg.SVM.gamma
        self.class_weight = class_weight if class_weight is not None else cfg.SVM.class_weight
        self.random_state = random_state if random_state is not None else cfg.RANDOM_STATE
        self.use_linear = use_linear if use_linear is not None else cfg.SVM.use_linear
        
        self.model = None
        self.best_params = None
        self.training_history = {}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'SVMTrainer':
        """
        Train the SVM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            self
        """
        log.info("Training SVM...")
        start_time = time.time()
        
        if self.use_linear:
            # CalibratedClassifierCV with cv=3 already trains LinearSVC internally
            base = LinearSVC(C=self.C, class_weight=self.class_weight, random_state=self.random_state, max_iter=10000)
            calibrator = CalibratedClassifierCV(base, cv=3)
            calibrator.fit(X_train, y_train)
            self.model = calibrator
        else:
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                class_weight=self.class_weight,
                random_state=self.random_state,
                probability=True  # Enable probability estimates
            )
            self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_prob = self.model.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
        
        self.training_history['train_time'] = train_time
        self.training_history['train_metrics'] = train_metrics
        
        log.info(f"Training complete in {train_time:.2f}s")
        log.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        log.info(f"Train F1: {train_metrics['f1_macro']:.4f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_prob = self.model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
            self.training_history['val_metrics'] = val_metrics
            
            log.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            log.info(f"Val F1: {val_metrics['f1_macro']:.4f}")
        
        return self
    
    def train_with_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict = None,
        cv: int = 5
    ) -> 'SVMTrainer':
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
            # Choose default based on linear vs non-linear
            param_grid = cfg.SVM.linear_param_grid if self.use_linear else cfg.SVM.param_grid
        
        log.info("Running GridSearchCV for SVM...")
        log.info(f"Parameter grid: {param_grid}")
        
        start_time = time.time()
        
        if self.use_linear:
            base_model = LinearSVC(class_weight=self.class_weight, random_state=self.random_state, max_iter=10000)
        else:
            base_model = SVC(class_weight=self.class_weight, random_state=self.random_state, probability=True)
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_splitter,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        search_time = time.time() - start_time
        
        log.info(f"\n GridSearchCV complete in {search_time:.2f}s")
        log.info(f"Best params: {self.best_params}")
        log.info(f"Best CV F1: {grid_search.best_score_:.4f}")
        
        self.training_history['grid_search_time'] = search_time
        self.training_history['best_params'] = self.best_params
        self.training_history['best_cv_score'] = grid_search.best_score_
        
        # If linear, calibrate the best linear estimator to obtain probabilities
        if self.use_linear:
            calibrator = CalibratedClassifierCV(self.model, cv=3)
            calibrator.fit(X_train, y_train)
            self.model = calibrator

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
        """Save the model using joblib (safer & faster for sklearn objects)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'training_history': self.training_history
        }, path)
        log.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SVMTrainer':
        """Load a saved model."""
        data = joblib.load(path)
        
        trainer = cls()
        trainer.model = data['model']
        trainer.best_params = data['best_params']
        trainer.training_history = data['training_history']
        return trainer


def main():
    """Main training function."""

    log.info("=" * 60)
    log.info("SVM TRAINING")
    log.info("=" * 60)

    # Paths
    features_path = os.path.join(cfg.PATHS.tfidf_dir, 'tfidf_features.pkl')
    model_dir = cfg.PATHS.svm_dir
    os.makedirs(model_dir, exist_ok=True)

    # Load features
    log.info("Loading TF-IDF features...")
    features = joblib.load(features_path)

    X_train = features['X_train']
    X_val = features['X_val']
    X_test = features['X_test']
    y_train = features['y_train']
    y_val = features['y_val']
    y_test = features['y_test']

    log.info("Train: %s", X_train.shape)
    log.info("Val: %s", X_val.shape)
    log.info("Test: %s", X_test.shape)

    # Initialize trainer (defaults pulled from cfg.SVM)
    trainer = SVMTrainer()

    # Train with GridSearchCV (param grid chosen inside trainer based on use_linear)
    log.info("-" * 60)
    trainer.train_with_grid_search(
        X_train, y_train,
        cv=cfg.SVM.cv_folds
    )

    # Evaluate on validation set
    log.info("-" * 60)
    log.info("Validation Results:")
    val_metrics = trainer.evaluate(X_val, y_val)
    print_metrics(val_metrics)

    # Evaluate on test set
    log.info("-" * 60)
    log.info("Test Results:")
    y_pred_test = trainer.predict(X_test)
    y_prob_test = trainer.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)
    print_metrics(test_metrics)

    # Save model
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    trainer.save(model_path)

    # Save results (metrics, predictions, experiment log)
    save_training_results(
        model_name='SVM',
        model_dir=model_dir,
        model_path=model_path,
        metrics_dict={
            'model': 'SVM',
            'use_linear': trainer.use_linear,
            'best_params': trainer.best_params,
            'validation': val_metrics,
            'test': test_metrics,
        },
        test_metrics=test_metrics,
        y_true=y_test,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
        experiment_config={'use_linear': trainer.use_linear, 'best_params': trainer.best_params},
    )

    return trainer, test_metrics


if __name__ == "__main__":
    trainer, metrics = main()
