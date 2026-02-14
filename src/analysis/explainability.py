"""
Explainability and Interpretability Analysis

Provides interpretability insights for model predictions:
1. TF-IDF feature importance for LR (top predictive words)
2. SVM feature analysis
3. Attention weight visualization for BiLSTM
4. Error categorization taxonomy
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def analyze_lr_feature_importance(top_n: int = 30) -> Dict:
    """
    Analyze Logistic Regression feature importance using model coefficients.
    
    Identifies the most predictive words for each class (Real vs Fake).
    
    Args:
        top_n: Number of top features to extract per class
        
    Returns:
        Dictionary with feature importance analysis
    """
    print("\n" + "="*60)
    print("1. LOGISTIC REGRESSION FEATURE IMPORTANCE")
    print("="*60)
    
    # Load LR model
    model_path = os.path.join(BASE_DIR, 'experiments', 'lr', 'lr_model.pkl')
    if not os.path.exists(model_path):
        print("LR model not found")
        return {}
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    lr_model = model_data['model']
    
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(BASE_DIR, 'data', 'features', 'tfidf', 'tfidf_vectorizer.pkl')
    if not os.path.exists(tfidf_path):
        # Try alternative path
        tfidf_path = os.path.join(BASE_DIR, 'data', 'features', 'tfidf', 'tfidf_features.pkl')
        if not os.path.exists(tfidf_path):
            print("TF-IDF vectorizer not found")
            return {}
        with open(tfidf_path, 'rb') as f:
            features = pickle.load(f)
        if 'vectorizer' not in features:
            print("Vectorizer not in features file")
            return {}
        vectorizer = features['vectorizer']
    else:
        with open(tfidf_path, 'rb') as f:
            vectorizer_data = pickle.load(f)
        if isinstance(vectorizer_data, dict) and 'vectorizer' in vectorizer_data:
            vectorizer = vectorizer_data['vectorizer']
        else:
            vectorizer = vectorizer_data
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    
    # Top features for Fake news (positive coefficient)
    fake_indices = np.argsort(coefficients)[-top_n:][::-1]
    fake_features = [(feature_names[i], round(coefficients[i], 4)) for i in fake_indices]
    
    # Top features for Real news (negative coefficient)
    real_indices = np.argsort(coefficients)[:top_n]
    real_features = [(feature_names[i], round(coefficients[i], 4)) for i in real_indices]
    
    print(f"\n  Top {top_n} Predictive Words for FAKE News:")
    for word, coef in fake_features[:15]:
        print(f"    {word:>25s}: {coef:+.4f}")
    
    print(f"\n  Top {top_n} Predictive Words for REAL News:")
    for word, coef in real_features[:15]:
        print(f"    {word:>25s}: {coef:+.4f}")
    
    return {
        'fake_news_features': fake_features,
        'real_news_features': real_features,
        'total_features': len(feature_names),
        'nonzero_features': int(np.sum(coefficients != 0))
    }


def analyze_error_categories() -> Dict:
    """
    Categorize prediction errors into meaningful taxonomy.
    
    Categories:
    - Short text: < 50 words
    - Medium text: 50-200 words  
    - Long text: > 200 words
    - High confidence errors: model was very confident but wrong
    - Low confidence errors: model was uncertain
    """
    print("\n" + "="*60)
    print("2. ERROR CATEGORIZATION TAXONOMY")
    print("="*60)
    
    # Load test data
    test_path = os.path.join(BASE_DIR, 'data', 'splits', 'test.csv')
    test_df = pd.read_csv(test_path)
    
    # Load predictions
    models = {
        'Logistic Regression': 'lr',
        'SVM': 'svm',
        'BiLSTM': 'bilstm',
        'PhoBERT': 'bert'
    }
    
    all_errors = {}
    
    for model_name, dir_name in models.items():
        pred_path = os.path.join(BASE_DIR, 'experiments', dir_name, 'predictions.pkl')
        if not os.path.exists(pred_path):
            continue
        
        with open(pred_path, 'rb') as f:
            preds = pickle.load(f)
        
        y_true = preds['y_true']
        y_pred = preds['y_pred']
        y_prob = preds.get('y_prob', None)
        
        errors_mask = y_pred != y_true
        error_indices = np.where(errors_mask)[0]
        
        # Categorize errors
        error_analysis = {
            'total_errors': int(np.sum(errors_mask)),
            'false_positives': int(np.sum((y_pred == 1) & (y_true == 0))),
            'false_negatives': int(np.sum((y_pred == 0) & (y_true == 1))),
            'by_text_length': {'short': 0, 'medium': 0, 'long': 0},
            'by_confidence': {'high_conf_error': 0, 'low_conf_error': 0}
        }
        
        for idx in error_indices:
            if idx < len(test_df):
                text = str(test_df.iloc[idx].get('text', ''))
                word_count = len(text.split())
                
                if word_count < 50:
                    error_analysis['by_text_length']['short'] += 1
                elif word_count < 200:
                    error_analysis['by_text_length']['medium'] += 1
                else:
                    error_analysis['by_text_length']['long'] += 1
            
            if y_prob is not None and idx < len(y_prob):
                prob = y_prob[idx]
                confidence = max(prob, 1 - prob)
                if confidence > 0.8:
                    error_analysis['by_confidence']['high_conf_error'] += 1
                else:
                    error_analysis['by_confidence']['low_conf_error'] += 1
        
        all_errors[model_name] = error_analysis
        
        print(f"\n  {model_name}:")
        print(f"    Total errors: {error_analysis['total_errors']}")
        print(f"    FP: {error_analysis['false_positives']}, FN: {error_analysis['false_negatives']}")
        print(f"    By length - Short: {error_analysis['by_text_length']['short']}, "
              f"Medium: {error_analysis['by_text_length']['medium']}, "
              f"Long: {error_analysis['by_text_length']['long']}")
        if y_prob is not None:
            print(f"    High-conf errors: {error_analysis['by_confidence']['high_conf_error']}, "
                  f"Low-conf errors: {error_analysis['by_confidence']['low_conf_error']}")
    
    return all_errors


def create_feature_importance_plot(feature_analysis: Dict, save_dir: str):
    """Create feature importance visualization."""
    if not feature_analysis:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Fake news features
    fake_features = feature_analysis['fake_news_features'][:20]
    words = [f[0] for f in fake_features]
    coefs = [f[1] for f in fake_features]
    
    axes[0].barh(range(len(words)), coefs, color='#d62728', alpha=0.8)
    axes[0].set_yticks(range(len(words)))
    axes[0].set_yticklabels(words, fontsize=9)
    axes[0].set_xlabel('Coefficient', fontsize=11)
    axes[0].set_title('Top Predictive Words for Fake News', fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()
    
    # Real news features
    real_features = feature_analysis['real_news_features'][:20]
    words = [f[0] for f in real_features]
    coefs = [abs(f[1]) for f in real_features]
    
    axes[1].barh(range(len(words)), coefs, color='#2ca02c', alpha=0.8)
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words, fontsize=9)
    axes[1].set_xlabel('|Coefficient|', fontsize=11)
    axes[1].set_title('Top Predictive Words for Real News', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'feature_importance.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Feature importance plot saved to {save_dir}")


def create_error_taxonomy_plot(error_analysis: Dict, save_dir: str):
    """Create error taxonomy visualization."""
    if not error_analysis:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Error type distribution (FP vs FN)
    models = list(error_analysis.keys())
    fp_counts = [error_analysis[m]['false_positives'] for m in models]
    fn_counts = [error_analysis[m]['false_negatives'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, fp_counts, width, label='False Positives', color='#ff7f0e', alpha=0.8)
    axes[0].bar(x + width/2, fn_counts, width, label='False Negatives', color='#1f77b4', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Error Type Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    
    # Error by text length
    categories = ['short', 'medium', 'long']
    bottom = np.zeros(len(models))
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    for cat, color in zip(categories, colors):
        counts = [error_analysis[m]['by_text_length'][cat] for m in models]
        axes[1].bar(models, counts, bottom=bottom, label=f'{cat.capitalize()} text', color=color, alpha=0.8)
        bottom += np.array(counts)
    
    axes[1].set_ylabel('Error Count', fontsize=11)
    axes[1].set_title('Errors by Text Length', fontsize=13, fontweight='bold')
    axes[1].legend()
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'error_taxonomy.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'error_taxonomy.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Error taxonomy plot saved to {save_dir}")


def main():
    """Run all explainability analyses."""
    
    print("="*60)
    print("EXPLAINABILITY & INTERPRETABILITY ANALYSIS")
    print("="*60)
    
    results = {}
    
    # 1. Feature importance
    feature_analysis = analyze_lr_feature_importance(top_n=30)
    results['feature_importance'] = feature_analysis
    
    # 2. Error categorization
    error_analysis = analyze_error_categories()
    results['error_taxonomy'] = error_analysis
    
    # Create visualizations
    figures_dir = os.path.join(BASE_DIR, 'results', 'figures', 'explainability')
    paper_figures_dir = os.path.join(BASE_DIR, 'paper', 'figures')
    
    create_feature_importance_plot(feature_analysis, figures_dir)
    create_feature_importance_plot(feature_analysis, paper_figures_dir)
    create_error_taxonomy_plot(error_analysis, figures_dir)
    create_error_taxonomy_plot(error_analysis, paper_figures_dir)
    
    # Save results
    results_dir = os.path.join(BASE_DIR, 'results', 'tables')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'explainability_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = main()
