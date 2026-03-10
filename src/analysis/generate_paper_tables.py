"""
Generate Publication-Quality Tables for Research Paper


Creates LaTeX tables suitable for academic publication.
"""

import os
import pandas as pd

from src.utils.common import load_all_metrics, load_csv
from config import cfg

from src.utils.logger import get_logger
log = get_logger(__name__)


def _fetch_all_metrics():
    """Load all metrics (thin wrapper around shared utility)."""
    return load_all_metrics()


def table1_dataset_statistics(save_path: str):
    """
    Table 1: Dataset Statistics
    """
    # Load data statistics
    train_df = load_csv(os.path.join(cfg.PATHS.splits_dir, 'train.csv'), required_columns=['text', 'label'])
    val_df = load_csv(os.path.join(cfg.PATHS.splits_dir, 'val.csv'), required_columns=['text', 'label'])
    test_df = load_csv(os.path.join(cfg.PATHS.splits_dir, 'test.csv'), required_columns=['text', 'label'])
    
    stats = []
    
    for split_name, df in [('Huấn luyện', train_df), ('Xác thực', val_df), ('Kiểm tra', test_df)]:
        total = len(df)
        real = len(df[df['label'] == 0])
        fake = len(df[df['label'] == 1])
        
        # Text statistics
        df['text_len'] = df['text'].astype(str).apply(lambda x: len(x.split()))
        avg_len = df['text_len'].mean()
        
        stats.append({
            'Split': split_name,
            'Total': total,
            'Real News': f"{real} ({real/total*100:.1f}\\%)",
            'Fake News': f"{fake} ({fake/total*100:.1f}\\%)",
            'Avg. Length': f"{avg_len:.1f}"
        })
    
    df_stats = pd.DataFrame(stats)
    
    # Generate LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{Thống kê tập dữ liệu}
\label{tab:dataset}
\begin{tabular}{lcccc}
\toprule
\textbf{Tập} & \textbf{Tổng} & \textbf{Tin thật} & \textbf{Tin giả} & \textbf{Độ dài TB} \\
\midrule
"""
    
    for _, row in df_stats.iterrows():
        latex += f"{row['Split']} & {row['Total']} & {row['Real News']} & {row['Fake News']} & {row['Avg. Length']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    log.info(f"Table 1 saved: {save_path}")
    return df_stats


def table2_model_comparison(metrics: dict, save_path: str):
    """
    Table 2: Model Performance Comparison
    """
    rows = []
    
    model_order = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    
    for model in model_order:
        if model not in metrics:
            continue
        test = metrics[model]['test']
        rows.append({
            'Model': model,
            'Accuracy': test['accuracy'],
            'Precision': test['precision_macro'],
            'Recall': test['recall_macro'],
            'F1-Score': test['f1_macro'],
            'ROC-AUC': test['roc_auc']
        })
    
    df = pd.DataFrame(rows)
    
    # Find best values for bolding
    best_idx = {col: df[col].idxmax() for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']}
    
    # Generate LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{So sánh hiệu suất các mô hình trên tập kiểm tra}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
\textbf{Mô hình} & \textbf{Độ chính xác} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{ROC-AUC} \\
\midrule
"""
    
    for idx, row in df.iterrows():
        line = f"{row['Model']}"
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            val = row[col]
            if idx == best_idx[col]:
                line += f" & \\textbf{{{val:.4f}}}"
            else:
                line += f" & {val:.4f}"
        line += " \\\\\n"
        latex += line
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    log.info(f"Table 2 saved: {save_path}")
    return df


def table3_per_class_metrics(metrics: dict, save_path: str):
    """
    Table 3: Per-Class Performance
    """
    rows = []
    
    model_order = ['Logistic Regression', 'SVM', 'BiLSTM', 'PhoBERT']
    
    for model in model_order:
        if model not in metrics:
            continue
        test = metrics[model]['test']
        
        # Real news
        rows.append({
            'Model': model,
            'Class': 'Thật',
            'Precision': test['precision_per_class'][0],
            'Recall': test['recall_per_class'][0],
            'F1-Score': test['f1_per_class'][0]
        })
        
        # Fake news
        rows.append({
            'Model': model,
            'Class': 'Giả',
            'Precision': test['precision_per_class'][1],
            'Recall': test['recall_per_class'][1],
            'F1-Score': test['f1_per_class'][1]
        })
    
    df = pd.DataFrame(rows)
    
    # Generate LaTeX with multirow
    latex = r"""
\begin{table}[h]
\centering
\caption{Chỉ số hiệu suất theo từng lớp}
\label{tab:perclass}
\begin{tabular}{llccc}
\toprule
\textbf{Mô hình} & \textbf{Lớp} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
"""
    
    current_model = None
    for idx, row in df.iterrows():
        if row['Model'] != current_model:
            if current_model is not None:
                latex += "\\midrule\n"
            current_model = row['Model']
            model_str = f"\\multirow{{2}}{{*}}{{{row['Model']}}}"
        else:
            model_str = ""
        
        latex += f"{model_str} & {row['Class']} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['F1-Score']:.4f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    log.info(f"Table 3 saved: {save_path}")
    return df


def table4_hyperparameters(save_path: str):
    """
    Table 4: Model Hyperparameters — all values pulled from cfg.
    """
    latex = rf"""
\begin{{table}}[h]
\centering
\caption{{Siêu tham số các mô hình}}
\label{{tab:hyperparams}}
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Mô hình}} & \textbf{{Siêu tham số}} \\
\midrule
Logistic Regression & C={cfg.LR.C}, class\_weight={cfg.LR.class_weight}, max\_iter={cfg.LR.max_iter} \\
\midrule
SVM & C={cfg.SVM.C}, kernel={cfg.SVM.kernel}, gamma={cfg.SVM.gamma} \\
\midrule
\multirow{{4}}{{*}}{{BiLSTM}} & Embedding dim={cfg.BILSTM.embedding_dim}, Hidden dim={cfg.BILSTM.hidden_dim} \\
 & Num layers={cfg.BILSTM.num_layers}, Dropout={cfg.BILSTM.dropout} \\
 & Learning rate={cfg.BILSTM.learning_rate}, Batch size={cfg.BILSTM.batch_size} \\
 & Early stopping patience={cfg.BILSTM.patience} \\
\midrule
\multirow{{4}}{{*}}{{PhoBERT}} & Pre-trained: {cfg.PHOBERT.model_name} \\
 & Max length={cfg.PHOBERT.max_seq_len}, Weight decay={cfg.PHOBERT.weight_decay} \\
 & Learning rate={cfg.PHOBERT.learning_rate}, Batch size={cfg.PHOBERT.batch_size} \\
 & Epochs={cfg.PHOBERT.epochs}, Warmup ratio={cfg.PHOBERT.warmup_ratio} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    log.info(f"Table 4 saved: {save_path}")


def table5_training_time(metrics: dict, save_path: str):
    """
    Table 5: Training Time Comparison
    """
    rows = []
    
    model_info = {
        'Logistic Regression': {'params': '~10K', 'type': 'ML truyền thống'},
        'SVM': {'params': '~10K', 'type': 'ML truyền thống'},
        'BiLSTM': {'params': '~500K', 'type': 'Học sâu'},
        'PhoBERT': {'params': '~135M', 'type': 'Transformer'}
    }
    
    for model, info in model_info.items():
        if model in metrics:
            train_time = metrics[model].get('training_time', 'N/A')
            if isinstance(train_time, (int, float)):
                if train_time < 60:
                    time_str = f"{train_time:.1f}s"
                else:
                    time_str = f"{train_time/60:.1f}min"
            else:
                time_str = str(train_time)
            
            rows.append({
                'Model': model,
                'Type': info['type'],
                'Parameters': info['params'],
                'Training Time': time_str
            })
    
    df = pd.DataFrame(rows)
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Độ phức tạp và thời gian huấn luyện}
\label{tab:complexity}
\begin{tabular}{llcc}
\toprule
\textbf{Mô hình} & \textbf{Loại} & \textbf{Tham số} & \textbf{Thời gian huấn luyện} \\
\midrule
"""
    
    for _, row in df.iterrows():
        latex += f"{row['Model']} & {row['Type']} & {row['Parameters']} & {row['Training Time']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    log.info(f"Table 5 saved: {save_path}")
    return df


def main():
    """Generate all publication tables."""


    print("="*70)
    print("GENERATING PUBLICATION-QUALITY TABLES")
    print("="*70)
    
    # Create output directory
    tables_dir = cfg.PATHS.paper_tables_dir
    os.makedirs(tables_dir, exist_ok=True)
    
    # Load metrics
    print("\n Loading metrics...")
    metrics = _fetch_all_metrics()
    
    # Generate tables
    print("\n Generating tables...")
    
    table1_dataset_statistics(os.path.join(tables_dir, 'table1_dataset.tex'))
    table2_model_comparison(metrics, os.path.join(tables_dir, 'table2_results.tex'))
    table3_per_class_metrics(metrics, os.path.join(tables_dir, 'table3_perclass.tex'))
    table4_hyperparameters(os.path.join(tables_dir, 'table4_hyperparams.tex'))
    table5_training_time(metrics, os.path.join(tables_dir, 'table5_complexity.tex'))
    
    print("\n" + "="*70)
    print("ALL TABLES GENERATED!")
    print("="*70)
    print(f"\n Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
