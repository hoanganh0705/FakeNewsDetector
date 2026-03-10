"""
Cross-Validation Evaluation for Traditional ML Models


Performs stratified k-fold cross-validation for Logistic Regression and SVM
to provide more robust performance estimates beyond a single train/test split.

This addresses reviewer concerns about single-seed evaluation and strengthens
the validity of reported results.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, make_scorer
)

from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


def get_scorers() -> Dict:
    """Define scoring metrics for cross-validation."""
    return {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'roc_auc': 'roc_auc',
    }


def run_cross_validation(
    model, 
    X: np.ndarray, 
    y: np.ndarray, 
    model_name: str,
    n_folds: int = None,
    n_seeds: int = 3,
    random_state: int = None
) -> Dict:
    """
    Run stratified k-fold cross-validation with multiple random seeds.
    
    Args:
        model: sklearn model (unfitted)
        X: Feature matrix
        y: Labels
        model_name: Name of the model
        n_folds: Number of CV folds
        n_seeds: Number of random seeds
        random_state: Base random seed
        
    Returns:
        Dictionary with CV results
    """
    if random_state is None:
        random_state = cfg.RANDOM_STATE
    if n_folds is None:
        n_folds = cfg.LR.cv_folds
    scorers = get_scorers()
    all_results = {metric: [] for metric in scorers.keys()}
    
    log.info(f"\n{'='*60}")
    log.info(f"Cross-Validation: {model_name}")
    log.info(f"{'='*60}")
    log.info(f"Folds: {n_folds}, Seeds: {n_seeds}")
    
    for seed_idx in range(n_seeds):
        seed = random_state + seed_idx * 100
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scorers,
            n_jobs=-1,
            return_train_score=False
        )
        
        for metric in scorers.keys():
            key = f'test_{metric}'
            all_results[metric].extend(cv_results[key].tolist())
        
        mean_f1 = np.mean(cv_results['test_f1_macro'])
        log.info(f"Seed {seed}: Mean F1 = {mean_f1:.4f}")
    
    # Compute summary statistics
    summary = {}
    for metric, scores in all_results.items():
        scores = np.array(scores)
        summary[metric] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'ci_lower': float(np.percentile(scores, 2.5)),
            'ci_upper': float(np.percentile(scores, 97.5)),
            'all_scores': [float(s) for s in scores]
        }
    
    log.info(f"\n  Summary ({n_folds * n_seeds} total folds):")
    log.info(f"    Accuracy:  {summary['accuracy']['mean']:.4f} +/- {summary['accuracy']['std']:.4f}")
    log.info(f"    F1-Score:  {summary['f1_macro']['mean']:.4f} +/- {summary['f1_macro']['std']:.4f}")
    log.info(f"    Precision: {summary['precision_macro']['mean']:.4f} +/- {summary['precision_macro']['std']:.4f}")
    log.info(f"    Recall:    {summary['recall_macro']['mean']:.4f} +/- {summary['recall_macro']['std']:.4f}")
    log.info(f"    ROC-AUC:   {summary['roc_auc']['mean']:.4f} +/- {summary['roc_auc']['std']:.4f}")
    
    return summary


def main():
    """Run cross-validation for LR and SVM."""


    cv_folds = cfg.LR.cv_folds
    cv_seeds = 3

    print("="*60)
    print("CROSS-VALIDATION EVALUATION")
    print("="*60)
    print(f"Running stratified {cv_folds}-fold CV with {cv_seeds} random seeds")
    print(f"Total: {cv_folds * cv_seeds} folds per model for robust estimates")
    
    # Load TF-IDF features (combine train + val for CV)
    features_path = os.path.join(cfg.PATHS.tfidf_dir, 'tfidf_features.pkl')
    
    if not os.path.exists(features_path):
        print(f"Features not found: {features_path}")
        return
    
    features = joblib.load(features_path)
    
    # Combine train and validation for cross-validation
    from scipy.sparse import vstack
    X_train = features['X_train']
    X_val = features['X_val']
    y_train = features['y_train']
    y_val = features['y_val']
    
    # Stack train + val for full CV
    X_cv = vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])
    
    print(f"\nCombined dataset: {X_cv.shape[0]} samples, {X_cv.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_cv.astype(int))}")
    
    results = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(
        C=cfg.LR.C, max_iter=cfg.LR.max_iter, solver='liblinear',
        class_weight=cfg.LR.class_weight, random_state=cfg.RANDOM_STATE, n_jobs=cfg.LR.n_jobs
    )
    results['Logistic Regression'] = run_cross_validation(
        lr_model, X_cv, y_cv, 'Logistic Regression',
        n_folds=cv_folds, n_seeds=cv_seeds
    )
    
    # 2. SVM
    svm_model = SVC(
        C=cfg.SVM.C, kernel=cfg.SVM.kernel, gamma=cfg.SVM.gamma,
        class_weight=cfg.SVM.class_weight, random_state=cfg.RANDOM_STATE, probability=True
    )
    results['SVM'] = run_cross_validation(
        svm_model, X_cv, y_cv, 'SVM',
        n_folds=cv_folds, n_seeds=cv_seeds
    )
    
    # Save results
    results_dir = cfg.PATHS.tables_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    cv_output = {
        'description': f'Stratified {cv_folds}-fold cross-validation with {cv_seeds} random seeds',
        'total_folds_per_model': cv_folds * cv_seeds,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for model_name, model_results in results.items():
        cv_output['results'][model_name] = {
            metric: {k: v for k, v in stats.items() if k != 'all_scores'}
            for metric, stats in model_results.items()
        }
    
    with open(os.path.join(results_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_output, f, indent=2)
    
    # Create comparison table
    rows = []
    for model_name, model_results in results.items():
        rows.append({
            'Model': model_name,
            'Accuracy': f"{model_results['accuracy']['mean']:.4f} +/- {model_results['accuracy']['std']:.4f}",
            'F1-Score': f"{model_results['f1_macro']['mean']:.4f} +/- {model_results['f1_macro']['std']:.4f}",
            'Precision': f"{model_results['precision_macro']['mean']:.4f} +/- {model_results['precision_macro']['std']:.4f}",
            'Recall': f"{model_results['recall_macro']['mean']:.4f} +/- {model_results['recall_macro']['std']:.4f}",
            'ROC-AUC': f"{model_results['roc_auc']['mean']:.4f} +/- {model_results['roc_auc']['std']:.4f}",
        })
    
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(os.path.join(results_dir, 'cross_validation_summary.csv'), index=False)
    
    # Save LaTeX table
    with open(os.path.join(results_dir, 'cross_validation.tex'), 'w') as f:
        f.write(f"% Kết quả kiểm chéo ({cv_folds}-fold, {cv_seeds} seed)\n")
        f.write(cv_df.to_latex(index=False, escape=True))
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
