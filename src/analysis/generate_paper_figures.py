"""
Generate Publication-Quality Figures for Research Paper


Creates high-resolution figures suitable for academic publication.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.utils.common import load_all_metrics, MODEL_DIR_MAP, load_csv
from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


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


def figure0a_overall_distribution(save_path: str):
    """
    Figure 0a: Overall Label Distribution (Donut Chart)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # --- Load data ---
    raw_df = load_csv(cfg.PATHS.raw_data, required_columns=['label'])

    class_labels = ['Thật (0)', 'Giả (1)']
    palette = ['#2E86AB', '#C73E1D']  # blue = real, red = fake

    counts = raw_df['label'].value_counts().sort_index().values  # [real, fake]
    total = counts.sum()
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=class_labels,
        autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct * total / 100)):,})',
        startangle=90,
        colors=palette,
        pctdistance=0.72,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        textprops={'fontsize': 11}
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title('Phân phối nhãn tổng thể', fontweight='bold', pad=14)
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'Tổng\n{total:,}', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"\u2705 Saved: {save_path}")


def figure0b_split_distribution(save_path: str):
    """
    Figure 0b: Per-Split Label Distribution (Grouped Bar Chart)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Load data ---
    splits = {}
    for name in ['train', 'val', 'test']:
        path = os.path.join(cfg.PATHS.splits_dir, f'{name}.csv')
        if os.path.exists(path):
            splits[name] = load_csv(path, required_columns=['label'])

    palette = ['#2E86AB', '#C73E1D']  # blue = real, red = fake

    split_names = ['Huấn luyện', 'Xác thực', 'Kiểm tra']
    split_keys  = ['train', 'val', 'test']
    real_counts = [int((splits[k]['label'] == 0).sum()) for k in split_keys if k in splits]
    fake_counts = [int((splits[k]['label'] == 1).sum()) for k in split_keys if k in splits]

    x = np.arange(len(split_names))
    width = 0.32

    bars_real = ax.bar(x - width / 2, real_counts, width,
                       label='Thật (0)', color=palette[0],
                       edgecolor='black', linewidth=0.5)
    bars_fake = ax.bar(x + width / 2, fake_counts, width,
                       label='Giả (1)', color=palette[1],
                       edgecolor='black', linewidth=0.5)

    # Value labels on bars
    for bars in (bars_real, bars_fake):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Số lượng mẫu')
    ax.set_xlabel('Tập dữ liệu')
    ax.set_title('Phân phối theo tập dữ liệu', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_ylim(0, max(real_counts + fake_counts) * 1.18)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"\u2705 Saved: {save_path}")


def load_all_data():
    """Load metrics and predictions."""
    metrics = load_all_metrics()
    predictions = {}

    for full_name, dir_name in MODEL_DIR_MAP.items():
        pred_path = os.path.join(cfg.PATHS.experiments_dir, dir_name, 'predictions.pkl')
        if os.path.exists(pred_path):
            predictions[full_name] = joblib.load(pred_path)

    return metrics, predictions


def figure1_model_comparison_bar(metrics: dict, save_path: str):
    """
    Figure 1: Model Performance Comparison (Bar Chart)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = list(MODEL_DIR_MAP.keys())
    metrics_list = ['Độ chính xác', 'Precision', 'Recall', 'F1-Score']
    
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
    
    ax.set_ylabel('Điểm số')
    ax.set_xlabel('Mô hình')
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
    log.info(f"Saved: {save_path}")


def figure2_confusion_matrices(metrics: dict, save_path: str):
    """
    Figure 2: Confusion Matrices Grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    
    models = list(MODEL_DIR_MAP.keys())
    class_names = ['Thật', 'Giả']
    
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
        axes[idx].set_xlabel('Nhãn dự đoán')
        axes[idx].set_ylabel('Nhãn thực tế')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(class_names)
        axes[idx].set_yticklabels(class_names)
        
        # Add accuracy
        acc = metrics[model]['test']['accuracy']
        axes[idx].text(0.5, -0.18, f'Độ chính xác: {acc:.2%}', 
                      transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Tỷ lệ chuẩn hóa')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"Saved: {save_path}")


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
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Ngẫu nhiên (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tỷ lệ dương tính giả (FPR)')
    ax.set_ylabel('Tỷ lệ dương tính thật (TPR)')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add annotation for best model
    ax.annotate('PhoBERT đạt\nAUC cao nhất', 
               xy=(0.1, 0.9), fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"Saved: {save_path}")


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
    ax.set_xlabel('Độ phủ (Recall)')
    ax.set_ylabel('Độ chính xác (Precision)')
    ax.legend(loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"Saved: {save_path}")


def figure5_per_class_performance(metrics: dict, save_path: str):
    """
    Figure 5: Per-Class Performance Comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(MODEL_DIR_MAP.keys())
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
    axes[0].set_ylabel('Điểm số')
    axes[0].set_xlabel('Mô hình')
    axes[0].set_title('Tin thật (Lớp 0)', fontweight='bold')
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
    axes[1].set_ylabel('Điểm số')
    axes[1].set_xlabel('Mô hình')
    axes[1].set_title('Tin giả (Lớp 1)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylim(0.75, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"Saved: {save_path}")


def figure6_model_paradigm_comparison(metrics: dict, save_path: str):
    """
    Figure 6: Performance by Model Paradigm
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group models by paradigm
    paradigms = {
        'ML truyền thống\n(TF-IDF)': ['Logistic Regression', 'SVM'],
        'Học sâu\n(Word Embeddings)': ['BiLSTM'],
        'Transformer\n(Mô hình ngôn ngữ)': ['PhoBERT']
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
    
    bars1 = ax.bar(x - width, paradigm_acc, width, label='Độ chính xác', color='#2E86AB', edgecolor='black')
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
    
    ax.set_ylabel('Điểm số')
    ax.set_xlabel('Phương pháp mô hình')
    ax.set_xticks(x)
    ax.set_xticklabels(paradigm_names)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add arrow showing improvement
    ax.annotate('', xy=(2.2, 0.96), xytext=(0.2, 0.88),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.2, 0.90, 'Cải thiện\nhiệu suất', ha='center', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf')
    plt.close()
    log.info(f"Saved: {save_path}")


def main():
    """Generate all publication figures."""


    print("="*70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*70)
    
    # Create output directory
    figures_dir = cfg.PATHS.paper_figures_dir
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    print("\n Loading data...")
    metrics, predictions = load_all_data()
    
    # Generate figures
    print("\n Generating figures...")
    
    figure0a_overall_distribution(
        os.path.join(figures_dir, 'fig0a_overall_distribution.png'))
    
    figure0b_split_distribution(
        os.path.join(figures_dir, 'fig0b_split_distribution.png'))
    
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
    print("ALL FIGURES GENERATED!")
    print("="*70)
    print(f"\n Figures saved to: {figures_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(figures_dir)):
        print(f"   - {f}")


if __name__ == "__main__":
    main()
