"""
Generate Publication-Quality Figures for Research Paper

Creates high-resolution figures suitable for academic publication.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def load_all_data():
    """Load metrics and predictions."""
    metrics = {}
    predictions = {}
    
    models = {
        'LR': ('lr', 'Logistic Regression'),
        'SVM': ('svm', 'SVM'),
        'BiLSTM': ('bilstm', 'BiLSTM'),
        'PhoBERT': ('bert', 'PhoBERT')
    }
    
    for short_name, (dir_name, full_name) in models.items():
        # Load metrics
        metrics_path = os.path.join(BASE_DIR, 'experiments', dir_name, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics[full_name] = json.load(f)
        
        # Load predictions
        pred_path = os.path.join(BASE_DIR, 'experiments', dir_name, 'predictions.pkl')
        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                predictions[full_name] = pickle.load(f)
    
    return metrics, predictions


def figure1_model_comparison_bar(metrics: dict, save_path: str):
    """
    Figure 1: Model Performance Comparison (Bar Chart)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    data = []
    for model in models:
        if model in metrics:
            test = metrics[model]['test']
            data.append([
                test['accuracy'],
                test['precision_macro'],
                test['recall_macro'],
                test['f1_macro']
            ])
    
    x = np.arange(len(models))
    width = 0.2
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric, color) in enumerate(zip(metrics_list, colors)):
        values = [data[j][i] for j in range(len(models))]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=color, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add horizontal line at best score
    best_f1 = max([data[j][3] for j in range(len(models))])
    ax.axhline(y=best_f1, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    print(f"✅ Saved: {save_path}")


def figure2_confusion_matrices(metrics: dict, save_path: str):
    """
    Figure 2: Confusion Matrices Grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    
    models = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    class_names = ['Real', 'Fake']
    
    for idx, model in enumerate(models):
        if model not in metrics:
            continue
            
        cm = np.array(metrics[model]['test']['confusion_matrix'])
        
        # Normalize for display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                axes[idx].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                             ha='center', va='center', color=color, fontsize=11)
        
        axes[idx].set_title(f'{model}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(class_names)
        axes[idx].set_yticklabels(class_names)
        
        # Add accuracy
        acc = metrics[model]['test']['accuracy']
        axes[idx].text(0.5, -0.18, f'Accuracy: {acc:.2%}', 
                      transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Count')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    print(f"✅ Saved: {save_path}")


def figure3_roc_curves(predictions: dict, save_path: str):
    """
    Figure 3: ROC Curves Comparison
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = {'Logistic Regression': '#2E86AB', 'SVM': '#A23B72', 
              'BiLSTM': '#F18F01', 'PhoBERT': '#C73E1D'}
    linestyles = {'Logistic Regression': '-', 'SVM': '--', 
                  'BiLSTM': '-.', 'PhoBERT': '-'}
    
    for model_name, preds in predictions.items():
        fpr, tpr, _ = roc_curve(preds['y_true'], preds['y_prob'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors.get(model_name, 'gray'),
               linestyle=linestyles.get(model_name, '-'),
               linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add annotation for best model
    ax.annotate('PhoBERT achieves\nhighest AUC', 
               xy=(0.1, 0.9), fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    print(f"✅ Saved: {save_path}")


def figure4_precision_recall_curves(predictions: dict, save_path: str):
    """
    Figure 4: Precision-Recall Curves
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = {'Logistic Regression': '#2E86AB', 'SVM': '#A23B72', 
              'BiLSTM': '#F18F01', 'PhoBERT': '#C73E1D'}
    
    for model_name, preds in predictions.items():
        precision, recall, _ = precision_recall_curve(preds['y_true'], preds['y_prob'])
        ap = average_precision_score(preds['y_true'], preds['y_prob'])
        
        ax.plot(recall, precision, color=colors.get(model_name, 'gray'),
               linewidth=2, label=f'{model_name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    print(f"✅ Saved: {save_path}")


def figure5_per_class_performance(metrics: dict, save_path: str):
    """
    Figure 5: Per-Class Performance Comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    x = np.arange(len(models))
    width = 0.35
    
    # Real news performance
    real_f1 = [metrics[m]['test']['f1_per_class'][0] for m in models if m in metrics]
    fake_f1 = [metrics[m]['test']['f1_per_class'][1] for m in models if m in metrics]
    
    # Plot Real News
    axes[0].bar(x - width/2, [metrics[m]['test']['precision_per_class'][0] for m in models], 
               width, label='Precision', color='#2E86AB', edgecolor='black', linewidth=0.5)
    axes[0].bar(x + width/2, [metrics[m]['test']['recall_per_class'][0] for m in models], 
               width, label='Recall', color='#F18F01', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Score')
    axes[0].set_xlabel('Model')
    axes[0].set_title('Real News (Class 0)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].set_ylim(0.75, 1.0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot Fake News
    axes[1].bar(x - width/2, [metrics[m]['test']['precision_per_class'][1] for m in models], 
               width, label='Precision', color='#2E86AB', edgecolor='black', linewidth=0.5)
    axes[1].bar(x + width/2, [metrics[m]['test']['recall_per_class'][1] for m in models], 
               width, label='Recall', color='#F18F01', edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Score')
    axes[1].set_xlabel('Model')
    axes[1].set_title('Fake News (Class 1)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylim(0.75, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    print(f"✅ Saved: {save_path}")


def figure6_model_paradigm_comparison(metrics: dict, save_path: str):
    """
    Figure 6: Performance by Model Paradigm
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group models by paradigm
    paradigms = {
        'Traditional ML\n(TF-IDF)': ['Logistic Regression', 'SVM'],
        'Deep Learning\n(Word Embeddings)': ['BiLSTM'],
        'Transformer\n(Pre-trained LM)': ['PhoBERT']
    }
    
    paradigm_names = list(paradigms.keys())
    paradigm_f1 = []
    paradigm_acc = []
    paradigm_auc = []
    
    for paradigm, models in paradigms.items():
        f1_scores = [metrics[m]['test']['f1_macro'] for m in models if m in metrics]
        acc_scores = [metrics[m]['test']['accuracy'] for m in models if m in metrics]
        auc_scores = [metrics[m]['test']['roc_auc'] for m in models if m in metrics]
        
        paradigm_f1.append(max(f1_scores))
        paradigm_acc.append(max(acc_scores))
        paradigm_auc.append(max(auc_scores))
    
    x = np.arange(len(paradigm_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, paradigm_acc, width, label='Accuracy', color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x, paradigm_f1, width, label='F1-Score', color='#F18F01', edgecolor='black')
    bars3 = ax.bar(x + width, paradigm_auc, width, label='ROC-AUC', color='#C73E1D', edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Model Paradigm')
    ax.set_xticks(x)
    ax.set_xticklabels(paradigm_names)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add arrow showing improvement
    ax.annotate('', xy=(2.2, 0.96), xytext=(0.2, 0.88),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.2, 0.90, 'Performance\nImprovement', ha='center', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    print(f"✅ Saved: {save_path}")


def main():
    """Generate all publication figures."""
    
    print("="*70)
    print("📊 GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*70)
    
    # Create output directory
    figures_dir = os.path.join(BASE_DIR, 'paper', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    print("\n📥 Loading data...")
    metrics, predictions = load_all_data()
    
    # Generate figures
    print("\n📈 Generating figures...")
    
    figure1_model_comparison_bar(
        metrics, os.path.join(figures_dir, 'fig1_model_comparison.png'))
    
    figure2_confusion_matrices(
        metrics, os.path.join(figures_dir, 'fig2_confusion_matrices.png'))
    
    figure3_roc_curves(
        predictions, os.path.join(figures_dir, 'fig3_roc_curves.png'))
    
    figure4_precision_recall_curves(
        predictions, os.path.join(figures_dir, 'fig4_pr_curves.png'))
    
    figure5_per_class_performance(
        metrics, os.path.join(figures_dir, 'fig5_per_class.png'))
    
    figure6_model_paradigm_comparison(
        metrics, os.path.join(figures_dir, 'fig6_paradigm_comparison.png'))
    
    print("\n" + "="*70)
    print("✅ ALL FIGURES GENERATED!")
    print("="*70)
    print(f"\n📁 Figures saved to: {figures_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(figures_dir)):
        print(f"   - {f}")


if __name__ == "__main__":
    main()
