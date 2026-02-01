"""
Error Analysis Script

Analyzes model errors to understand:
1. What types of news are misclassified
2. Common patterns in errors
3. Differences between models
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def load_test_data_with_predictions():
    """Load test data and predictions from all models."""
    # Load test data
    test_path = os.path.join(BASE_DIR, 'data', 'splits', 'test.csv')
    test_df = pd.read_csv(test_path)
    
    # Load predictions
    models = ['lr', 'svm', 'bilstm', 'bert']
    model_names = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    
    for model, name in zip(models, model_names):
        pred_path = os.path.join(BASE_DIR, 'experiments', model, 'predictions.pkl')
        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                preds = pickle.load(f)
            test_df[f'{name}_pred'] = preds['y_pred']
            test_df[f'{name}_prob'] = preds['y_prob']
            test_df[f'{name}_correct'] = (test_df['label'] == preds['y_pred']).astype(int)
    
    return test_df


def analyze_error_patterns(test_df: pd.DataFrame, model_name: str):
    """Analyze error patterns for a specific model."""
    pred_col = f'{model_name}_pred'
    correct_col = f'{model_name}_correct'
    
    if pred_col not in test_df.columns:
        return None
    
    # Get errors
    errors_df = test_df[test_df[correct_col] == 0].copy()
    correct_df = test_df[test_df[correct_col] == 1].copy()
    
    # Calculate text length
    errors_df['text_len'] = errors_df['text'].astype(str).apply(len)
    correct_df['text_len'] = correct_df['text'].astype(str).apply(len)
    
    # Error analysis
    analysis = {
        'total_errors': len(errors_df),
        'error_rate': len(errors_df) / len(test_df),
        'false_positives': len(errors_df[(errors_df['label'] == 0) & (errors_df[pred_col] == 1)]),
        'false_negatives': len(errors_df[(errors_df['label'] == 1) & (errors_df[pred_col] == 0)]),
        'avg_text_len_errors': errors_df['text_len'].mean(),
        'avg_text_len_correct': correct_df['text_len'].mean(),
    }
    
    return analysis, errors_df


def plot_error_analysis(test_df: pd.DataFrame, save_dir: str):
    """Create visualizations for error analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    available_models = [m for m in model_names if f'{m}_pred' in test_df.columns]
    
    # 1. Error rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    error_rates = []
    
    for model in available_models:
        correct_col = f'{model}_correct'
        error_rate = 1 - test_df[correct_col].mean()
        error_rates.append(error_rate)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(available_models)]
    bars = ax.bar(available_models, error_rates, color=colors)
    
    for bar, rate in zip(bars, error_rates):
        ax.annotate(f'{rate:.2%}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=11)
    
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Model Error Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(error_rates) * 1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved error rate comparison")
    
    # 2. False Positives vs False Negatives
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fp_rates = []
    fn_rates = []
    
    for model in available_models:
        pred_col = f'{model}_pred'
        fp = len(test_df[(test_df['label'] == 0) & (test_df[pred_col] == 1)])
        fn = len(test_df[(test_df['label'] == 1) & (test_df[pred_col] == 0)])
        fp_rates.append(fp)
        fn_rates.append(fn)
    
    x = np.arange(len(available_models))
    width = 0.35
    
    ax.bar(x - width/2, fp_rates, width, label='False Positives (Real → Fake)', color='#ff7f0e')
    ax.bar(x + width/2, fn_rates, width, label='False Negatives (Fake → Real)', color='#1f77b4')
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('False Positives vs False Negatives', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_models)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fp_fn_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved FP/FN comparison")
    
    # 3. Text length distribution for errors vs correct
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, model in enumerate(available_models[:4]):
        correct_col = f'{model}_correct'
        
        test_df['text_len'] = test_df['text'].astype(str).apply(lambda x: len(x.split()))
        
        errors = test_df[test_df[correct_col] == 0]['text_len']
        correct = test_df[test_df[correct_col] == 1]['text_len']
        
        axes[idx].hist(correct, bins=30, alpha=0.7, label='Correct', color='#2ca02c')
        axes[idx].hist(errors, bins=30, alpha=0.7, label='Errors', color='#d62728')
        axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Text Length (words)')
        axes[idx].set_ylabel('Count')
        axes[idx].legend()
    
    plt.suptitle('Text Length Distribution: Errors vs Correct Predictions', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'text_length_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved text length analysis")
    
    # 4. Agreement between models
    if len(available_models) >= 2:
        # Count how many models agree on each prediction
        test_df['model_agreement'] = 0
        for model in available_models:
            pred_col = f'{model}_pred'
            test_df['model_agreement'] += (test_df[pred_col] == test_df['label']).astype(int)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agreement_counts = test_df['model_agreement'].value_counts().sort_index()
        ax.bar(agreement_counts.index, agreement_counts.values, color='#1f77b4')
        
        ax.set_xlabel('Number of Models Correct', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Model Agreement Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(available_models) + 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_agreement.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved model agreement analysis")


def find_hard_examples(test_df: pd.DataFrame, n: int = 10):
    """Find examples that all models got wrong."""
    model_names = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    available_models = [m for m in model_names if f'{m}_pred' in test_df.columns]
    
    # Count errors per sample
    test_df['total_errors'] = 0
    for model in available_models:
        correct_col = f'{model}_correct'
        test_df['total_errors'] += (1 - test_df[correct_col])
    
    # Get hardest examples (all models wrong)
    hardest = test_df[test_df['total_errors'] == len(available_models)]
    
    print(f"\n📋 Found {len(hardest)} examples that ALL models got wrong:")
    
    for idx, row in hardest.head(n).iterrows():
        print(f"\n{'='*50}")
        print(f"ID: {row['id']}")
        print(f"True Label: {'Fake' if row['label'] == 1 else 'Real'}")
        print(f"Text (first 200 chars): {str(row['text'])[:200]}...")
    
    return hardest


def main():
    """Run error analysis."""
    
    print("="*60)
    print("🔍 ERROR ANALYSIS")
    print("="*60)
    
    # Load data with predictions
    print("\n📥 Loading test data with predictions...")
    test_df = load_test_data_with_predictions()
    
    print(f"   Total test samples: {len(test_df)}")
    
    # Analyze each model
    print("\n📊 Error Analysis by Model:")
    print("-"*60)
    
    model_names = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    
    for model in model_names:
        result = analyze_error_patterns(test_df, model)
        if result:
            analysis, errors_df = result
            print(f"\n{model}:")
            print(f"   Total Errors: {analysis['total_errors']} ({analysis['error_rate']:.2%})")
            print(f"   False Positives: {analysis['false_positives']}")
            print(f"   False Negatives: {analysis['false_negatives']}")
            print(f"   Avg Text Length (Errors): {analysis['avg_text_len_errors']:.0f}")
            print(f"   Avg Text Length (Correct): {analysis['avg_text_len_correct']:.0f}")
    
    # Generate visualizations
    figures_dir = os.path.join(BASE_DIR, 'results', 'figures', 'error_analysis')
    print(f"\n📈 Generating error analysis visualizations...")
    plot_error_analysis(test_df, figures_dir)
    
    # Find hard examples
    print("\n" + "="*60)
    hardest = find_hard_examples(test_df, n=5)
    
    # Save error analysis results
    tables_dir = os.path.join(BASE_DIR, 'results', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Save hard examples
    hardest_path = os.path.join(tables_dir, 'hard_examples.csv')
    if len(hardest) > 0:
        hardest[['id', 'title', 'text', 'label']].to_csv(hardest_path, index=False)
        print(f"\n✅ Hard examples saved to {hardest_path}")
    
    print("\n" + "="*60)
    print("✅ ERROR ANALYSIS COMPLETE!")
    print("="*60)
    
    return test_df


if __name__ == "__main__":
    test_df = main()
