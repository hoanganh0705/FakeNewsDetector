"""
Statistical Significance Testing for Model Comparison

Performs:
1. McNemar's test for comparing classifier predictions
2. Bootstrap confidence intervals
3. Effect size calculations
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two classifiers.
    
    Returns:
        chi2_stat: Chi-squared statistic
        p_value: P-value for the test
    """
    # Build contingency table
    # b: model1 correct, model2 wrong
    # c: model1 wrong, model2 correct
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    
    # McNemar's test with continuity correction
    if b + c == 0:
        return 0.0, 1.0
    
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    return chi2_stat, p_value


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, 
                                   metric_func, n_bootstrap: int = 1000, 
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Returns:
        mean: Mean of bootstrap samples
        lower: Lower bound of CI
        upper: Upper bound of CI
    """
    n_samples = len(y_true)
    bootstrap_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    mean = np.mean(bootstrap_scores)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    
    return mean, lower, upper


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_predictions() -> Dict[str, dict]:
    """Load predictions from all models."""
    predictions = {}
    
    models = {
        'Logistic Regression': 'lr',
        'SVM': 'svm', 
        'BiLSTM': 'bilstm',
        'PhoBERT': 'bert'
    }
    
    for name, dir_name in models.items():
        pred_path = os.path.join(BASE_DIR, 'experiments', dir_name, 'predictions.pkl')
        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                predictions[name] = pickle.load(f)
    
    return predictions


def run_statistical_analysis():
    """Run comprehensive statistical analysis."""
    
    print("="*70)
    print("📊 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*70)
    
    # Load predictions
    predictions = load_predictions()
    
    if len(predictions) < 2:
        print("❌ Need at least 2 models for comparison")
        return
    
    # Get ground truth
    y_true = list(predictions.values())[0]['y_true']
    
    # 1. McNemar's Test - Pairwise Comparison
    print("\n" + "="*70)
    print("1. McNEMAR'S TEST (Pairwise Classifier Comparison)")
    print("="*70)
    print("H0: The two classifiers have equal error rates")
    print("Significance level: α = 0.05")
    print("-"*70)
    
    model_names = list(predictions.keys())
    mcnemar_results = []
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            y_pred1 = predictions[model1]['y_pred']
            y_pred2 = predictions[model2]['y_pred']
            
            chi2, p_value = mcnemar_test(y_true, y_pred1, y_pred2)
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            result = {
                'Model 1': model1,
                'Model 2': model2,
                'Chi-squared': chi2,
                'p-value': p_value,
                'Significant': significance
            }
            mcnemar_results.append(result)
            
            print(f"\n{model1} vs {model2}:")
            print(f"   χ² = {chi2:.4f}, p = {p_value:.4f} {significance}")
    
    mcnemar_df = pd.DataFrame(mcnemar_results)
    
    # 2. Bootstrap Confidence Intervals
    print("\n" + "="*70)
    print("2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("="*70)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    ci_results = []
    
    for model_name, preds in predictions.items():
        y_pred = preds['y_pred']
        
        # Accuracy CI
        acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score
        )
        
        # F1 CI
        f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(
            y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average='macro')
        )
        
        ci_results.append({
            'Model': model_name,
            'Accuracy': f"{acc_mean:.4f}",
            'Acc 95% CI': f"[{acc_lower:.4f}, {acc_upper:.4f}]",
            'F1-Score': f"{f1_mean:.4f}",
            'F1 95% CI': f"[{f1_lower:.4f}, {f1_upper:.4f}]"
        })
        
        print(f"\n{model_name}:")
        print(f"   Accuracy: {acc_mean:.4f} (95% CI: [{acc_lower:.4f}, {acc_upper:.4f}])")
        print(f"   F1-Score: {f1_mean:.4f} (95% CI: [{f1_lower:.4f}, {f1_upper:.4f}])")
    
    ci_df = pd.DataFrame(ci_results)
    
    # 3. Effect Size (Cohen's d) for PhoBERT vs others
    print("\n" + "="*70)
    print("3. EFFECT SIZE ANALYSIS (PhoBERT vs Others)")
    print("="*70)
    print("Cohen's d interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,")
    print("                          0.5-0.8 = medium, > 0.8 = large")
    print("-"*70)
    
    if 'PhoBERT' in predictions:
        phobert_correct = (predictions['PhoBERT']['y_pred'] == y_true).astype(float)
        
        for model_name, preds in predictions.items():
            if model_name != 'PhoBERT':
                other_correct = (preds['y_pred'] == y_true).astype(float)
                d = cohens_d(phobert_correct, other_correct)
                
                effect = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                
                print(f"\nPhoBERT vs {model_name}:")
                print(f"   Cohen's d = {d:.4f} ({effect} effect)")
    
    # Save results
    results_dir = os.path.join(BASE_DIR, 'results', 'tables')
    os.makedirs(results_dir, exist_ok=True)
    
    mcnemar_df.to_csv(os.path.join(results_dir, 'mcnemar_test.csv'), index=False)
    ci_df.to_csv(os.path.join(results_dir, 'confidence_intervals.csv'), index=False)
    
    # Save LaTeX version
    with open(os.path.join(results_dir, 'statistical_tests.tex'), 'w') as f:
        f.write("% McNemar's Test Results\n")
        f.write(mcnemar_df.to_latex(index=False, escape=False))
        f.write("\n\n% Confidence Intervals\n")
        f.write(ci_df.to_latex(index=False, escape=False))
    
    print("\n" + "="*70)
    print("✅ STATISTICAL ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    
    return mcnemar_df, ci_df


if __name__ == "__main__":
    mcnemar_df, ci_df = run_statistical_analysis()
