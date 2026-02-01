"""
Generate Publication-Quality Tables for Research Paper

Creates LaTeX tables suitable for academic publication.
"""

import os
import sys
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def load_metrics():
    """Load all metrics."""
    metrics = {}
    models = {
        'Logistic Regression': 'lr',
        'SVM': 'svm',
        'BiLSTM': 'bilstm',
        'PhoBERT': 'bert'
    }
    
    for name, dir_name in models.items():
        path = os.path.join(BASE_DIR, 'experiments', dir_name, 'metrics.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics[name] = json.load(f)
    
    return metrics


def table1_dataset_statistics(save_path: str):
    """
    Table 1: Dataset Statistics
    """
    # Load data statistics
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'splits', 'train.csv'))
    val_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'splits', 'val.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'splits', 'test.csv'))
    
    stats = []
    
    for split_name, df in [('Training', train_df), ('Validation', val_df), ('Test', test_df)]:
        total = len(df)
        real = len(df[df['label'] == 0])
        fake = len(df[df['label'] == 1])
        
        # Text statistics
        df['text_len'] = df['text'].astype(str).apply(lambda x: len(x.split()))
        avg_len = df['text_len'].mean()
        
        stats.append({
            'Split': split_name,
            'Total': total,
            'Real News': f"{real} ({real/total*100:.1f}%)",
            'Fake News': f"{fake} ({fake/total*100:.1f}%)",
            'Avg. Length': f"{avg_len:.1f}"
        })
    
    df_stats = pd.DataFrame(stats)
    
    # Generate LaTeX
    latex = r"""
\begin{table}[h]
\centering
\caption{Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{lcccc}
\toprule
\textbf{Split} & \textbf{Total} & \textbf{Real News} & \textbf{Fake News} & \textbf{Avg. Length} \\
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
    
    print(f"✅ Table 1 saved: {save_path}")
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
\caption{Model Performance Comparison on Test Set}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{ROC-AUC} \\
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
    
    print(f"✅ Table 2 saved: {save_path}")
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
            'Class': 'Real',
            'Precision': test['precision_per_class'][0],
            'Recall': test['recall_per_class'][0],
            'F1-Score': test['f1_per_class'][0]
        })
        
        # Fake news
        rows.append({
            'Model': model,
            'Class': 'Fake',
            'Precision': test['precision_per_class'][1],
            'Recall': test['recall_per_class'][1],
            'F1-Score': test['f1_per_class'][1]
        })
    
    df = pd.DataFrame(rows)
    
    # Generate LaTeX with multirow
    latex = r"""
\begin{table}[h]
\centering
\caption{Per-Class Performance Metrics}
\label{tab:perclass}
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
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
    
    print(f"✅ Table 3 saved: {save_path}")
    return df


def table4_hyperparameters(save_path: str):
    """
    Table 4: Model Hyperparameters
    """
    latex = r"""
\begin{table}[h]
\centering
\caption{Model Hyperparameters}
\label{tab:hyperparams}
\begin{tabular}{ll}
\toprule
\textbf{Model} & \textbf{Hyperparameters} \\
\midrule
Logistic Regression & C=10, solver=lbfgs, max\_iter=1000 \\
\midrule
SVM & C=10, kernel=rbf, gamma=scale \\
\midrule
\multirow{4}{*}{BiLSTM} & Embedding dim=128, Hidden dim=128 \\
 & Num layers=2, Dropout=0.3 \\
 & Learning rate=0.001, Batch size=32 \\
 & Early stopping patience=3 \\
\midrule
\multirow{4}{*}{PhoBERT} & Pre-trained: vinai/phobert-base \\
 & Max length=256, Hidden dim=768 \\
 & Learning rate=2e-5, Batch size=16 \\
 & Epochs=5, Warmup steps=100 \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ Table 4 saved: {save_path}")


def table5_training_time(metrics: dict, save_path: str):
    """
    Table 5: Training Time Comparison
    """
    rows = []
    
    model_info = {
        'Logistic Regression': {'params': '~10K', 'type': 'Traditional ML'},
        'SVM': {'params': '~10K', 'type': 'Traditional ML'},
        'BiLSTM': {'params': '~500K', 'type': 'Deep Learning'},
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
\caption{Model Complexity and Training Time}
\label{tab:complexity}
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Type} & \textbf{Parameters} & \textbf{Training Time} \\
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
    
    print(f"✅ Table 5 saved: {save_path}")
    return df


def main():
    """Generate all publication tables."""
    
    print("="*70)
    print("📊 GENERATING PUBLICATION-QUALITY TABLES")
    print("="*70)
    
    # Create output directory
    tables_dir = os.path.join(BASE_DIR, 'paper', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Load metrics
    print("\n📥 Loading metrics...")
    metrics = load_metrics()
    
    # Generate tables
    print("\n📋 Generating tables...")
    
    table1_dataset_statistics(os.path.join(tables_dir, 'table1_dataset.tex'))
    table2_model_comparison(metrics, os.path.join(tables_dir, 'table2_results.tex'))
    table3_per_class_metrics(metrics, os.path.join(tables_dir, 'table3_perclass.tex'))
    table4_hyperparameters(os.path.join(tables_dir, 'table4_hyperparams.tex'))
    table5_training_time(metrics, os.path.join(tables_dir, 'table5_complexity.tex'))
    
    print("\n" + "="*70)
    print("✅ ALL TABLES GENERATED!")
    print("="*70)
    print(f"\n📁 Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
