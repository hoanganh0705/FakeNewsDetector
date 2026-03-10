"""
Explainability and Interpretability Analysis


Provides interpretability insights for model predictions:
1. TF-IDF feature importance for LR (top predictive words)
2. SVM feature analysis
4. Error categorization taxonomy
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict

from src.utils.common import MODEL_DIR_MAP, load_csv
from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


def analyze_lr_feature_importance(top_n: int = 30) -> Dict:
    """
    Analyze Logistic Regression feature importance using model coefficients.
    
    Identifies the most predictive words for each class (Real vs Fake).
    
    Args:
        top_n: Number of top features to extract per class
        
    Returns:
        Dictionary with feature importance analysis
    """
    log.info("\n" + "="*60)
    log.info("1. LOGISTIC REGRESSION FEATURE IMPORTANCE")
    log.info("="*60)
    
    # Load LR model
    model_path = os.path.join(cfg.PATHS.lr_dir, 'lr_model.pkl')
    if not os.path.exists(model_path):
        log.info("LR model not found")
        return {}
    
    model_data = joblib.load(model_path)
    lr_model = model_data['model']
    
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(cfg.PATHS.tfidf_dir, 'tfidf_vectorizer.pkl')
    if not os.path.exists(tfidf_path):
        # Try alternative path
        tfidf_path = os.path.join(cfg.PATHS.tfidf_dir, 'tfidf_features.pkl')
        if not os.path.exists(tfidf_path):
            log.info("TF-IDF vectorizer not found")
            return {}
        features = joblib.load(tfidf_path)
        if 'vectorizer' not in features:
            log.info("Vectorizer not in features file")
            return {}
        vectorizer = features['vectorizer']
    else:
        vectorizer_data = joblib.load(tfidf_path)
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
    
    log.info(f"\n  Top {top_n} Predictive Words for FAKE News:")
    for word, coef in fake_features[:15]:
        log.info(f"    {word:>25s}: {coef:+.4f}")
    
    log.info(f"\n  Top {top_n} Predictive Words for REAL News:")
    for word, coef in real_features[:15]:
        log.info(f"    {word:>25s}: {coef:+.4f}")
    
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
    log.info("\n" + "="*60)
    log.info("2. ERROR CATEGORIZATION TAXONOMY")
    log.info("="*60)
    
    # Load test data
    test_path = os.path.join(cfg.PATHS.splits_dir, 'test.csv')
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    # Load predictions
    models = MODEL_DIR_MAP
    
    all_errors = {}
    
    for model_name, dir_name in models.items():
        pred_path = os.path.join(cfg.PATHS.experiments_dir, dir_name, 'predictions.pkl')
        if not os.path.exists(pred_path):
            continue
        
        preds = joblib.load(pred_path)

        y_true = np.asarray(preds['y_true']).reshape(-1)
        y_pred = np.asarray(preds['y_pred']).reshape(-1)
        y_prob = preds.get('y_prob', None)
        if y_prob is not None:
            y_prob = np.asarray(y_prob).reshape(-1)

        n = min(len(y_true), len(y_pred))
        if n == 0:
            log.warning("%s: Empty predictions, skipping", model_name)
            continue

        if len(y_true) != len(y_pred):
            log.warning(
                "%s: Length mismatch y_true=%d, y_pred=%d; truncating to %d",
                model_name,
                len(y_true),
                len(y_pred),
                n,
            )

        y_true = y_true[:n]
        y_pred = y_pred[:n]
        if y_prob is not None:
            y_prob = y_prob[:n]

        errors_mask = (y_pred != y_true)
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
        
        log.info(f"\n  {model_name}:")
        log.info(f"    Total errors: {error_analysis['total_errors']}")
        log.info(f"    FP: {error_analysis['false_positives']}, FN: {error_analysis['false_negatives']}")
        log.info(f"    By length - Short: {error_analysis['by_text_length']['short']}, "
              f"Medium: {error_analysis['by_text_length']['medium']}, "
              f"Long: {error_analysis['by_text_length']['long']}")
        if y_prob is not None:
            log.info(f"    High-conf errors: {error_analysis['by_confidence']['high_conf_error']}, "
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
    axes[0].set_xlabel('Hệ số', fontsize=11)
    axes[0].set_title('Từ dự đoán hàng đầu cho Tin giả', fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()
    
    # Real news features
    real_features = feature_analysis['real_news_features'][:20]
    words = [f[0] for f in real_features]
    coefs = [abs(f[1]) for f in real_features]
    
    axes[1].barh(range(len(words)), coefs, color='#2ca02c', alpha=0.8)
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words, fontsize=9)
    axes[1].set_xlabel('|Hệ số|', fontsize=11)
    axes[1].set_title('Từ dự đoán hàng đầu cho Tin thật', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'feature_importance.pdf'), bbox_inches='tight')
    plt.close(fig)
    log.info(f"\n  Feature importance plot saved to {save_dir}")


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
    
    axes[0].bar(x - width/2, fp_counts, width, label='Dương tính giả', color='#ff7f0e', alpha=0.8)
    axes[0].bar(x + width/2, fn_counts, width, label='Âm tính giả', color='#1f77b4', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    axes[0].set_ylabel('Số lượng', fontsize=11)
    axes[0].set_title('Phân phối loại lỗi', fontsize=13, fontweight='bold')
    axes[0].legend()
    
    # Error by text length
    categories = ['short', 'medium', 'long']
    bottom = np.zeros(len(models))
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    cat_labels = {'short': 'Văn bản ngắn', 'medium': 'Văn bản vừa', 'long': 'Văn bản dài'}
    
    for cat, color in zip(categories, colors):
        counts = [error_analysis[m]['by_text_length'][cat] for m in models]
        axes[1].bar(models, counts, bottom=bottom, label=cat_labels[cat], color=color, alpha=0.8)
        bottom += np.array(counts)
    
    axes[1].set_ylabel('Số lỗi', fontsize=11)
    axes[1].set_title('Lỗi theo độ dài văn bản', fontsize=13, fontweight='bold')
    axes[1].legend()
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'error_taxonomy.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'error_taxonomy.pdf'), bbox_inches='tight')
    plt.close(fig)
    log.info(f"Error taxonomy plot saved to {save_dir}")


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
    figures_dir = os.path.join(cfg.PATHS.figures_dir, 'explainability')
    paper_figures_dir = cfg.PATHS.paper_figures_dir
    
    create_feature_importance_plot(feature_analysis, figures_dir)
    create_feature_importance_plot(feature_analysis, paper_figures_dir)
    create_error_taxonomy_plot(error_analysis, figures_dir)
    create_error_taxonomy_plot(error_analysis, paper_figures_dir)
    
    # Save results
    results_dir = cfg.PATHS.tables_dir
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'explainability_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    results = main()
