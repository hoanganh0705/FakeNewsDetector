"""
Statistical Significance Testing for Model Comparison

Performs:
1. McNemar's test for comparing classifier predictions
2. Holm-Bonferroni correction for multiple comparisons
3. Bootstrap confidence intervals (10,000 iterations)
4. Effect size calculations (Cohen's d)
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
    
    Uses continuity correction: chi2 = (|b - c| - 1)^2 / (b + c)
    
    Args:
        y_true: Ground truth labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
    
    Returns:
        chi2_stat: Chi-squared statistic
        p_value: P-value for the test
    """
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


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[dict]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    This is a step-down procedure that is more powerful than standard
    Bonferroni while still controlling the family-wise error rate (FWER).
    
    Args:
        p_values: List of raw p-values
        alpha: Significance level
        
    Returns:
        List of dicts with original and adjusted p-values and significance
    """
    n = len(p_values)
    indexed_pvalues = sorted(enumerate(p_values), key=lambda x: x[1])
    
    results = [None] * n
    
    for rank, (orig_idx, p) in enumerate(indexed_pvalues):
        adjusted_alpha = alpha / (n - rank)
        adjusted_p = min(p * (n - rank), 1.0)
        
        results[orig_idx] = {
            'original_p': p,
            'adjusted_p': adjusted_p,
            'adjusted_alpha': adjusted_alpha,
            'significant': p < adjusted_alpha
        }
    
    # Enforce monotonicity: adjusted p-values should be non-decreasing
    sorted_results = [results[idx] for idx, _ in indexed_pvalues]
    for i in range(1, len(sorted_results)):
        sorted_results[i]['adjusted_p'] = max(
            sorted_results[i]['adjusted_p'], 
            sorted_results[i-1]['adjusted_p']
        )
    
    for rank, (orig_idx, _) in enumerate(indexed_pvalues):
        results[orig_idx]['adjusted_p'] = sorted_results[rank]['adjusted_p']
        results[orig_idx]['significant'] = sorted_results[rank]['adjusted_p'] < alpha
    
    return results


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, 
                                   metric_func, n_bootstrap: int = 10000, 
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Uses 10,000 bootstrap iterations (modern standard for reporting).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_func: Metric function(y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap iterations (default: 10000)
        confidence: Confidence level (default: 0.95)
    
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
    """
    Calculate Cohen's d effect size.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
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
    """Run comprehensive statistical analysis with multiple comparison correction."""
    
    print("="*70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*70)
    
    predictions = load_predictions()
    
    if len(predictions) < 2:
        print("Need at least 2 models for comparison")
        return
    
    y_true = list(predictions.values())[0]['y_true']
    
    # ================================================================
    # 1. McNemar's Test with Holm-Bonferroni Correction
    # ================================================================
    print("\n" + "="*70)
    print("1. McNEMAR'S TEST (Pairwise Classifier Comparison)")
    print("="*70)
    print("H0: The two classifiers have equal error rates")
    print("Significance level: alpha = 0.05")
    print("Correction: Holm-Bonferroni for multiple comparisons")
    print("-"*70)
    
    model_names = list(predictions.keys())
    mcnemar_results = []
    raw_p_values = []
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            y_pred1 = predictions[model1]['y_pred']
            y_pred2 = predictions[model2]['y_pred']
            
            chi2, p_value = mcnemar_test(y_true, y_pred1, y_pred2)
            
            raw_p_values.append(p_value)
            mcnemar_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'Chi-squared': chi2,
                'p-value (raw)': p_value,
            })
    
    # Apply Holm-Bonferroni correction
    corrections = holm_bonferroni_correction(raw_p_values)
    
    for i, result in enumerate(mcnemar_results):
        result['p-value (adjusted)'] = corrections[i]['adjusted_p']
        result['Significant (raw)'] = "Yes" if result['p-value (raw)'] < 0.05 else "No"
        result['Significant (corrected)'] = "Yes" if corrections[i]['significant'] else "No"
        
        sig_raw = "*" if result['p-value (raw)'] < 0.05 else "ns"
        sig_adj = "*" if corrections[i]['significant'] else "ns"
        
        print(f"\n{result['Model 1']} vs {result['Model 2']}:")
        print(f"   chi2 = {result['Chi-squared']:.4f}")
        print(f"   p (raw)      = {result['p-value (raw)']:.4f} [{sig_raw}]")
        print(f"   p (adjusted) = {result['p-value (adjusted)']:.4f} [{sig_adj}]")
    
    mcnemar_df = pd.DataFrame(mcnemar_results)
    
    print("\n" + "-"*70)
    print(f"NOTE: Holm-Bonferroni correction applied across {len(raw_p_values)} comparisons")
    print(f"to control family-wise error rate (FWER).")
    
    # ================================================================
    # 2. Bootstrap Confidence Intervals (10,000 iterations)
    # ================================================================
    print("\n" + "="*70)
    print("2. BOOTSTRAP CONFIDENCE INTERVALS (95%, n=10,000)")
    print("="*70)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    ci_results = []
    
    for model_name, preds in predictions.items():
        y_pred = preds['y_pred']
        
        acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score, n_bootstrap=10000
        )
        
        f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(
            y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average='macro'),
            n_bootstrap=10000
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
    
    # ================================================================
    # 3. Effect Size (Cohen's d)
    # ================================================================
    print("\n" + "="*70)
    print("3. EFFECT SIZE ANALYSIS (Cohen's d)")
    print("="*70)
    print("Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,")
    print("                0.5-0.8 = medium, > 0.8 = large")
    print("-"*70)
    
    effect_sizes = []
    
    if 'PhoBERT' in predictions:
        phobert_correct = (predictions['PhoBERT']['y_pred'] == y_true).astype(float)
        
        for model_name, preds in predictions.items():
            if model_name != 'PhoBERT':
                other_correct = (preds['y_pred'] == y_true).astype(float)
                d = cohens_d(phobert_correct, other_correct)
                
                effect = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                
                effect_sizes.append({
                    'Comparison': f'PhoBERT vs {model_name}',
                    'Cohen_d': round(d, 4),
                    'Effect': effect
                })
                
                print(f"\nPhoBERT vs {model_name}:")
                print(f"   Cohen's d = {d:.4f} ({effect} effect)")
    
    effect_df = pd.DataFrame(effect_sizes) if effect_sizes else pd.DataFrame()
    
    # ================================================================
    # Save results
    # ================================================================
    results_dir = os.path.join(BASE_DIR, 'results', 'tables')
    os.makedirs(results_dir, exist_ok=True)
    
    mcnemar_df.to_csv(os.path.join(results_dir, 'mcnemar_test.csv'), index=False)
    ci_df.to_csv(os.path.join(results_dir, 'confidence_intervals.csv'), index=False)
    if len(effect_sizes) > 0:
        effect_df.to_csv(os.path.join(results_dir, 'effect_sizes.csv'), index=False)
    
    # Save LaTeX
    with open(os.path.join(results_dir, 'statistical_tests.tex'), 'w') as f:
        f.write("% McNemar's Test Results (with Holm-Bonferroni Correction)\n")
        f.write(mcnemar_df.to_latex(index=False, escape=False))
        f.write("\n\n% Bootstrap Confidence Intervals (n=10,000)\n")
        f.write(ci_df.to_latex(index=False, escape=False))
        if len(effect_sizes) > 0:
            f.write("\n\n% Effect Sizes (Cohen's d)\n")
            f.write(effect_df.to_latex(index=False, escape=False))
    
    # Save summary JSON
    summary = {
        'mcnemar_tests': mcnemar_results,
        'bootstrap_ci': ci_results,
        'effect_sizes': effect_sizes,
        'correction_method': 'Holm-Bonferroni',
        'bootstrap_iterations': 10000,
        'significance_level': 0.05,
        'num_comparisons': len(raw_p_values)
    }
    
    with open(os.path.join(results_dir, 'statistical_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    
    return mcnemar_df, ci_df, effect_df


if __name__ == "__main__":
    mcnemar_df, ci_df, effect_df = run_statistical_analysis()
