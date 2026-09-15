import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.features.embedding_features import TextDataset, collate_fn
from src.features.phobert_features import PhoBertDataset
from config import cfg
from src.utils.common import load_all_metrics, MODEL_DIR_MAP

from src.utils.logger import get_logger
log = get_logger(__name__)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")


__all__ = [
    "create_comparison_table",
    "create_per_class_table",
    "plot_model_comparison",
    "plot_confusion_matrices_grid",
    "plot_roc_curves_comparison",
    "plot_training_history",
    "generate_latex_table",
    "save_predictions_for_analysis",
]


def create_comparison_table(metrics: Dict[str, dict]) -> pd.DataFrame:
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
    rows = []
    
    class_names = ['Thật (0)', 'Giả (1)']
    
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
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Mô hình', fontsize=12)
    ax.set_ylabel('Điểm số', fontsize=12)
    ax.set_title('So sánh hiệu suất các mô hình', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved comparison chart to {save_path}")


def plot_confusion_matrices_grid(metrics: Dict[str, dict], save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    class_names = ['Thật', 'Giả']
    
    for idx, (model_name, model_metrics) in enumerate(metrics.items()):
        if idx >= 4:
            break
            
        cm = np.array(model_metrics['test']['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar=False, annot_kws={'size': 14})
        
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Dự đoán', fontsize=10)
        axes[idx].set_ylabel('Thực tế', fontsize=10)
        
        acc = model_metrics['test']['accuracy']
        axes[idx].text(0.5, -0.15, f'Độ chính xác: {acc:.4f}', 
                      transform=axes[idx].transAxes, ha='center', fontsize=10)
    
    plt.suptitle('So sánh ma trận nhầm lẫn', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved confusion matrices to {save_path}")


def plot_roc_curves_comparison(save_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    from sklearn.metrics import roc_curve, auc
    
    for (model_name, dir_name), color in zip(MODEL_DIR_MAP.items(), colors):
        pred_path = os.path.join(cfg.PATHS.experiments_dir, dir_name, 'predictions.pkl')
        
        if os.path.exists(pred_path):
            preds = joblib.load(pred_path)
            
            fpr, tpr, _ = roc_curve(preds['y_true'], preds['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Ngẫu nhiên (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tỷ lệ dương tính giả (FPR)', fontsize=12)
    ax.set_ylabel('Tỷ lệ dương tính thật (TPR)', fontsize=12)
    ax.set_title('So sánh đường cong ROC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved ROC curves to {save_path}")


def plot_training_history(save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BiLSTM history
    bilstm_metrics_path = os.path.join(cfg.PATHS.bilstm_dir, 'metrics.json')
    phobert_metrics_path = os.path.join(cfg.PATHS.bert_dir, 'metrics.json')
    
    models_data = []
    
    bilstm_model_path = os.path.join(cfg.PATHS.bilstm_dir, 'bilstm_model.pt')
    if os.path.exists(bilstm_model_path):
        checkpoint = torch.load(bilstm_model_path, map_location='cpu', weights_only=True)
        if 'training_history' in checkpoint:
            models_data.append(('BiLSTM', checkpoint['training_history']))
    
    phobert_model_path = os.path.join(cfg.PATHS.bert_dir, 'phobert_model.pt')
    if os.path.exists(phobert_model_path):
        checkpoint = torch.load(phobert_model_path, map_location='cpu', weights_only=True)
        if 'training_history' in checkpoint:
            models_data.append(('PhoBERT', checkpoint['training_history']))
    
    if not models_data:
        log.warning("No training history found for deep learning models")
        plt.close()
        return
    
    colors = {'BiLSTM': '#2ca02c', 'PhoBERT': '#d62728'}
    
    for model_name, history in models_data:
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], '--', color=colors[model_name], 
                        label=f'{model_name} Huấn luyện', alpha=0.7)
            axes[0].plot(epochs, history['val_loss'], '-', color=colors[model_name], 
                        label=f'{model_name} Xác thực')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mất mát', fontsize=12)
    axes[0].set_title('Mất mát huấn luyện & xác thực', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    for model_name, history in models_data:
        if 'val_f1' in history:
            epochs = range(1, len(history['val_f1']) + 1)
            axes[1].plot(epochs, history['val_f1'], '-o', color=colors[model_name], 
                        label=f'{model_name}', markersize=4)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score trên tập xác thực', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Tiến trình huấn luyện học sâu', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved training history to {save_path}")


def generate_latex_table(df: pd.DataFrame, save_path: str):
    df_formatted = df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f'{x:.4f}')
    
    latex = df_formatted.to_latex(index=False, escape=False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(latex)
    
    log.info(f"Saved LaTeX table to {save_path}")


def save_predictions_for_analysis():
    log.info("Checking predictions for all models...")

    tfidf_path = os.path.join(cfg.PATHS.tfidf_dir, 'tfidf_features.pkl')
    if not os.path.exists(tfidf_path):
        log.warning("TF-IDF features not found at %s", tfidf_path)
        log.info("Run Step 3 first: python src/features/extract_all_features.py")
        return
    tfidf_features = joblib.load(tfidf_path)
    
    X_test = tfidf_features['X_test']
    y_test = tfidf_features['y_test']
    
    lr_pred_path = os.path.join(cfg.PATHS.lr_dir, 'predictions.pkl')
    lr_model_path = os.path.join(cfg.PATHS.lr_dir, 'lr_model.pkl')
    if os.path.exists(lr_pred_path):
        log.info("  LR predictions already saved")
    elif os.path.exists(lr_model_path):
        lr_data = joblib.load(lr_model_path)
        lr_model = lr_data['model']
        
        y_pred = lr_model.predict(X_test)
        y_prob = lr_model.predict_proba(X_test)[:, 1]
        
        pred_path = os.path.join(cfg.PATHS.lr_dir, 'predictions.pkl')
        joblib.dump({'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob}, pred_path)
        log.info("\u2705 LR predictions saved")
    
    svm_pred_path = os.path.join(cfg.PATHS.svm_dir, 'predictions.pkl')
    svm_model_path = os.path.join(cfg.PATHS.svm_dir, 'svm_model.pkl')
    if os.path.exists(svm_pred_path):
        log.info("  SVM predictions already saved")
    elif os.path.exists(svm_model_path):
        svm_data = joblib.load(svm_model_path)
        svm_model = svm_data['model']
        
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)[:, 1]
        
        pred_path = os.path.join(cfg.PATHS.svm_dir, 'predictions.pkl')
        joblib.dump({'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob}, pred_path)
        log.info("SVM predictions saved")
    
    embedding_path = os.path.join(cfg.PATHS.embedding_dir, 'embedding_features.pkl')
    bilstm_model_path = os.path.join(cfg.PATHS.bilstm_dir, 'bilstm_model.pt')
    bilstm_pred_path = os.path.join(cfg.PATHS.bilstm_dir, 'predictions.pkl')

    if os.path.exists(bilstm_pred_path):
        log.info("  BiLSTM predictions already saved")
    elif os.path.exists(bilstm_model_path) and os.path.exists(embedding_path):
        emb_features = joblib.load(embedding_path)
        
        from src.training.train_bilstm import BiLSTMTrainer
        
        trainer = BiLSTMTrainer.load(bilstm_model_path)
        
        test_dataset = TextDataset(emb_features['test_sequences'], emb_features['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=cfg.BILSTM.batch_size, shuffle=False, collate_fn=collate_fn)
        
        y_pred, y_prob = trainer.predict(test_loader)
        
        pred_path = os.path.join(cfg.PATHS.bilstm_dir, 'predictions.pkl')
        joblib.dump({'y_true': emb_features['y_test'], 'y_pred': y_pred, 'y_prob': y_prob}, pred_path)
        log.info("BiLSTM predictions saved")
    
    phobert_path = os.path.join(cfg.PATHS.phobert_dir, 'phobert_features.pkl')
    phobert_model_path = os.path.join(cfg.PATHS.bert_dir, 'phobert_model.pt')
    phobert_pred_path = os.path.join(cfg.PATHS.bert_dir, 'predictions.pkl')

    if os.path.exists(phobert_pred_path):
        log.info("  PhoBERT predictions already saved")
    elif os.path.exists(phobert_model_path) and os.path.exists(phobert_path):
        phobert_features = joblib.load(phobert_path)
        
        from src.training.train_phobert import PhoBertTrainer
        
        trainer = PhoBertTrainer.load(phobert_model_path)
        
        test_dataset = PhoBertDataset(
            phobert_features['test_input_ids'],
            phobert_features['test_attention_mask'],
            phobert_features['y_test']
        )
        test_loader = DataLoader(test_dataset, batch_size=cfg.PHOBERT.batch_size, shuffle=False)
        
        y_pred, y_prob = trainer.predict(test_loader)
        
        pred_path = os.path.join(cfg.PATHS.bert_dir, 'predictions.pkl')
        joblib.dump({'y_true': phobert_features['y_test'], 'y_pred': y_pred, 'y_prob': y_prob}, pred_path)
        log.info("PhoBERT predictions saved")


def main():

    log.info("=" * 60)
    log.info("COMPREHENSIVE MODEL EVALUATION")
    log.info("=" * 60)

    figures_dir = cfg.PATHS.figures_dir
    tables_dir = cfg.PATHS.tables_dir
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    log.info("Loading model metrics...")
    metrics = load_all_metrics()

    if not metrics:
        log.warning("No metrics found. Please train models first.")
        return

    save_predictions_for_analysis()
    
    log.info("Creating comparison tables...")
    comparison_df = create_comparison_table(metrics)

    log.info("=" * 60)
    log.info("MODEL COMPARISON (Test Set)")
    log.info("=" * 60)
    log.info("\n%s", comparison_df.to_string(index=False))
    
    comparison_df.to_csv(os.path.join(tables_dir, 'model_comparison.csv'), index=False)
    generate_latex_table(comparison_df, os.path.join(tables_dir, 'model_comparison.tex'))
    
    per_class_df = create_per_class_table(metrics)
    per_class_df.to_csv(os.path.join(tables_dir, 'per_class_metrics.csv'), index=False)

    log.info("Per-Class Metrics:")
    log.info("\n%s", per_class_df.to_string(index=False))

    log.info("Generating visualizations...")
    
    plot_model_comparison(comparison_df, os.path.join(figures_dir, 'model_comparison.png'))
    
    plot_confusion_matrices_grid(metrics, os.path.join(figures_dir, 'confusion_matrices.png'))
    
    plot_roc_curves_comparison(os.path.join(figures_dir, 'roc_curves.png'))
    
    plot_training_history(os.path.join(figures_dir, 'training_history.png'))
    
    log.info("=" * 60)
    log.info("SUMMARY STATISTICS")
    log.info("=" * 60)

    best_model = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    best_acc = comparison_df.iloc[0]['Accuracy']

    log.info("Best Model: %s", best_model)
    log.info("  - Accuracy: %.4f", best_acc)
    log.info("  - F1-Score: %.4f", best_f1)
    
    lr_rows = comparison_df[comparison_df['Model'] == 'Logistic Regression']['F1-Score']
    if len(lr_rows) > 0 and best_model != 'Logistic Regression':
        baseline_f1 = lr_rows.values[0]
        improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
        log.info("Improvement over LR baseline: +%.1f%%", improvement)
    else:
        improvement = 0.0
    
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
    
    log.info("=" * 60)
    log.info("EVALUATION COMPLETE!")
    log.info("=" * 60)
    log.info("Results saved to:")
    log.info("  - Figures: %s", figures_dir)
    log.info("  - Tables: %s", tables_dir)
    
    return comparison_df, metrics


if __name__ == "__main__":
    comparison_df, metrics = main()
