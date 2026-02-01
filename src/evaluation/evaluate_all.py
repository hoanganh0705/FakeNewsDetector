"""
Comprehensive Evaluation Script for All Models

This script:
1. Loads all trained models
2. Evaluates on test set
3. Generates comparison tables
4. Creates visualizations (confusion matrices, ROC curves, etc.)
5. Performs statistical analysis
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.evaluation.metrics import (
    compute_metrics, 
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_precision_recall_curve
)
from src.features.embedding_features import TextDataset, collate_fn
from src.features.phobert_features import PhoBertDataset


# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_all_metrics() -> Dict[str, dict]:
    """Load metrics from all trained models."""
    metrics = {}
    
    model_dirs = {
        'Logistic Regression': 'lr',
        'SVM': 'svm',
        'BiLSTM': 'bilstm',
        'PhoBERT': 'bert'
    }
    
    for model_name, dir_name in model_dirs.items():
        metrics_path = os.path.join(BASE_DIR, 'experiments', dir_name, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics[model_name] = json.load(f)
            print(f"✅ Loaded metrics for {model_name}")
        else:
            print(f"⚠️  Metrics not found for {model_name}")
    
    return metrics


def create_comparison_table(metrics: Dict[str, dict]) -> pd.DataFrame:
    """Create a comparison table of all models."""
    rows = []
    
    for model_name, model_metrics in metrics.items():
        test = model_metrics.get('test', {})
        row = {
            'Model': model_name,
            'Accuracy': test.get('accuracy', 0),
            'Precision': test.get('precision_macro', 0),
            'Recall': test.get('recall_macro', 0),
            'F1-Score': test.get('f1_macro', 0),
            'ROC-AUC': test.get('roc_auc', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('F1-Score', ascending=False)
    
    return df


def create_per_class_table(metrics: Dict[str, dict]) -> pd.DataFrame:
    """Create per-class performance table."""
    rows = []
    
    class_names = ['Real (0)', 'Fake (1)']
    
    for model_name, model_metrics in metrics.items():
        test = model_metrics.get('test', {})
        
        for i, class_name in enumerate(class_names):
            row = {
                'Model': model_name,
                'Class': class_name,
                'Precision': test.get('precision_per_class', [0, 0])[i],
                'Recall': test.get('recall_per_class', [0, 0])[i],
                'F1-Score': test.get('f1_per_class', [0, 0])[i],
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_model_comparison(df: pd.DataFrame, save_path: str):
    """Create bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison chart to {save_path}")


def plot_confusion_matrices_grid(metrics: Dict[str, dict], save_path: str):
    """Plot confusion matrices for all models in a grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    class_names = ['Real', 'Fake']
    
    for idx, (model_name, model_metrics) in enumerate(metrics.items()):
        if idx >= 4:
            break
            
        cm = np.array(model_metrics['test']['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar=False, annot_kws={'size': 14})
        
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        
        # Add accuracy annotation
        acc = model_metrics['test']['accuracy']
        axes[idx].text(0.5, -0.15, f'Accuracy: {acc:.4f}', 
                      transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrices to {save_path}")


def plot_roc_curves_comparison(save_path: str):
    """Plot ROC curves for all models on the same graph."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_configs = [
        ('Logistic Regression', 'lr', 'tfidf'),
        ('SVM', 'svm', 'tfidf'),
        ('BiLSTM', 'bilstm', 'embedding'),
        ('PhoBERT', 'bert', 'phobert')
    ]
    
    from sklearn.metrics import roc_curve, auc
    
    for (model_name, dir_name, feature_type), color in zip(model_configs, colors):
        # Load predictions if available
        pred_path = os.path.join(BASE_DIR, 'experiments', dir_name, 'predictions.pkl')
        
        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                preds = pickle.load(f)
            
            fpr, tpr, _ = roc_curve(preds['y_true'], preds['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curves to {save_path}")


def plot_training_history(save_path: str):
    """Plot training history for deep learning models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BiLSTM history
    bilstm_metrics_path = os.path.join(BASE_DIR, 'experiments', 'bilstm', 'metrics.json')
    phobert_metrics_path = os.path.join(BASE_DIR, 'experiments', 'bert', 'metrics.json')
    
    models_data = []
    
    # Try to load training history from model files
    bilstm_model_path = os.path.join(BASE_DIR, 'experiments', 'bilstm', 'bilstm_model.pt')
    if os.path.exists(bilstm_model_path):
        checkpoint = torch.load(bilstm_model_path, map_location='cpu')
        if 'training_history' in checkpoint:
            models_data.append(('BiLSTM', checkpoint['training_history']))
    
    phobert_model_path = os.path.join(BASE_DIR, 'experiments', 'bert', 'phobert_model.pt')
    if os.path.exists(phobert_model_path):
        checkpoint = torch.load(phobert_model_path, map_location='cpu')
        if 'training_history' in checkpoint:
            models_data.append(('PhoBERT', checkpoint['training_history']))
    
    if not models_data:
        print("⚠️  No training history found for deep learning models")
        plt.close()
        return
    
    colors = {'BiLSTM': '#2ca02c', 'PhoBERT': '#d62728'}
    
    # Plot loss
    for model_name, history in models_data:
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], '--', color=colors[model_name], 
                        label=f'{model_name} Train', alpha=0.7)
            axes[0].plot(epochs, history['val_loss'], '-', color=colors[model_name], 
                        label=f'{model_name} Val')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot F1
    for model_name, history in models_data:
        if 'val_f1' in history:
            epochs = range(1, len(history['val_f1']) + 1)
            axes[1].plot(epochs, history['val_f1'], '-o', color=colors[model_name], 
                        label=f'{model_name}', markersize=4)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Deep Learning Training Progress', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training history to {save_path}")


def generate_latex_table(df: pd.DataFrame, save_path: str):
    """Generate LaTeX table for paper."""
    # Format numbers
    df_formatted = df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f'{x:.4f}')
    
    latex = df_formatted.to_latex(index=False, escape=False)
    
    # Add best result highlighting
    # Find best values and bold them
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved LaTeX table to {save_path}")


def save_predictions_for_analysis():
    """Save predictions from all models for further analysis."""
    print("\n📊 Generating predictions for all models...")
    
    # Load test features
    # TF-IDF features for LR and SVM
    tfidf_path = os.path.join(BASE_DIR, 'data', 'features', 'tfidf', 'tfidf_features.pkl')
    with open(tfidf_path, 'rb') as f:
        tfidf_features = pickle.load(f)
    
    X_test = tfidf_features['X_test']
    y_test = tfidf_features['y_test']
    
    # Logistic Regression
    lr_model_path = os.path.join(BASE_DIR, 'experiments', 'lr', 'lr_model.pkl')
    if os.path.exists(lr_model_path):
        with open(lr_model_path, 'rb') as f:
            lr_data = pickle.load(f)
        lr_model = lr_data['model']
        
        y_pred = lr_model.predict(X_test)
        y_prob = lr_model.predict_proba(X_test)[:, 1]
        
        pred_path = os.path.join(BASE_DIR, 'experiments', 'lr', 'predictions.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump({'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob}, f)
        print("  ✅ LR predictions saved")
    
    # SVM
    svm_model_path = os.path.join(BASE_DIR, 'experiments', 'svm', 'svm_model.pkl')
    if os.path.exists(svm_model_path):
        with open(svm_model_path, 'rb') as f:
            svm_data = pickle.load(f)
        svm_model = svm_data['model']
        
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)[:, 1]
        
        pred_path = os.path.join(BASE_DIR, 'experiments', 'svm', 'predictions.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump({'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob}, f)
        print("  ✅ SVM predictions saved")
    
    # BiLSTM
    embedding_path = os.path.join(BASE_DIR, 'data', 'features', 'embedding', 'embedding_features.pkl')
    bilstm_model_path = os.path.join(BASE_DIR, 'experiments', 'bilstm', 'bilstm_model.pt')
    
    if os.path.exists(bilstm_model_path) and os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            emb_features = pickle.load(f)
        
        from src.training.train_bilstm import BiLSTMTrainer
        
        trainer = BiLSTMTrainer.load(bilstm_model_path)
        
        test_dataset = TextDataset(emb_features['test_sequences'], emb_features['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        y_pred, y_prob = trainer.predict(test_loader)
        
        pred_path = os.path.join(BASE_DIR, 'experiments', 'bilstm', 'predictions.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump({'y_true': emb_features['y_test'], 'y_pred': y_pred, 'y_prob': y_prob}, f)
        print("  ✅ BiLSTM predictions saved")
    
    # PhoBERT
    phobert_path = os.path.join(BASE_DIR, 'data', 'features', 'phobert', 'phobert_features.pkl')
    phobert_model_path = os.path.join(BASE_DIR, 'experiments', 'bert', 'phobert_model.pt')
    
    if os.path.exists(phobert_model_path) and os.path.exists(phobert_path):
        with open(phobert_path, 'rb') as f:
            phobert_features = pickle.load(f)
        
        from src.training.train_phobert import PhoBertTrainer
        
        trainer = PhoBertTrainer.load(phobert_model_path)
        
        test_dataset = PhoBertDataset(
            phobert_features['test_input_ids'],
            phobert_features['test_attention_mask'],
            phobert_features['y_test']
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        y_pred, y_prob = trainer.predict(test_loader)
        
        pred_path = os.path.join(BASE_DIR, 'experiments', 'bert', 'predictions.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump({'y_true': phobert_features['y_test'], 'y_pred': y_pred, 'y_prob': y_prob}, f)
        print("  ✅ PhoBERT predictions saved")


def main():
    """Run comprehensive evaluation."""
    
    print("="*60)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Output directories
    figures_dir = os.path.join(BASE_DIR, 'results', 'figures')
    tables_dir = os.path.join(BASE_DIR, 'results', 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # 1. Load all metrics
    print("\n📥 Loading model metrics...")
    metrics = load_all_metrics()
    
    if not metrics:
        print("❌ No metrics found. Please train models first.")
        return
    
    # 2. Save predictions for ROC curves
    save_predictions_for_analysis()
    
    # 3. Create comparison table
    print("\n📋 Creating comparison tables...")
    comparison_df = create_comparison_table(metrics)
    
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON (Test Set)")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Save tables
    comparison_df.to_csv(os.path.join(tables_dir, 'model_comparison.csv'), index=False)
    generate_latex_table(comparison_df, os.path.join(tables_dir, 'model_comparison.tex'))
    
    # Per-class table
    per_class_df = create_per_class_table(metrics)
    per_class_df.to_csv(os.path.join(tables_dir, 'per_class_metrics.csv'), index=False)
    
    print("\n📋 Per-Class Metrics:")
    print(per_class_df.to_string(index=False))
    
    # 4. Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # Model comparison bar chart
    plot_model_comparison(comparison_df, os.path.join(figures_dir, 'model_comparison.png'))
    
    # Confusion matrices
    plot_confusion_matrices_grid(metrics, os.path.join(figures_dir, 'confusion_matrices.png'))
    
    # ROC curves
    plot_roc_curves_comparison(os.path.join(figures_dir, 'roc_curves.png'))
    
    # Training history
    plot_training_history(os.path.join(figures_dir, 'training_history.png'))
    
    # 5. Summary statistics
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    
    # Find best model
    best_model = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    best_acc = comparison_df.iloc[0]['Accuracy']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   - Accuracy: {best_acc:.4f}")
    print(f"   - F1-Score: {best_f1:.4f}")
    
    # Improvement over baseline
    baseline_f1 = comparison_df[comparison_df['Model'] == 'Logistic Regression']['F1-Score'].values[0]
    improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
    print(f"\n📈 Improvement over LR baseline: +{improvement:.1f}%")
    
    # Save summary
    summary = {
        'best_model': best_model,
        'best_accuracy': float(best_acc),
        'best_f1': float(best_f1),
        'improvement_over_baseline': float(improvement),
        'all_results': comparison_df.to_dict('records'),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(tables_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved to:")
    print(f"   - Figures: {figures_dir}")
    print(f"   - Tables: {tables_dir}")
    
    return comparison_df, metrics


if __name__ == "__main__":
    comparison_df, metrics = main()
