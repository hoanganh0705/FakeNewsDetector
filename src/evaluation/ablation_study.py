"""
Ablation Study for Vietnamese Fake News Detection


Evaluates the impact of different components on model performance:
1. TF-IDF vocabulary size impact (LR)
2. N-gram range impact (LR)
3. Word segmentation impact (LR)
4. BiLSTM architecture variations
5. PhoBERT max sequence length impact
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from config import cfg
from src.utils.common import load_csv

from src.utils.logger import get_logger
log = get_logger(__name__)


def load_text_data() -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Load train and test text data."""
    train_path = os.path.join(cfg.PATHS.splits_dir, 'train.csv')
    test_path = os.path.join(cfg.PATHS.splits_dir, 'test.csv')
    
    train_df = load_csv(train_path, required_columns=['text', 'label'])
    test_df = load_csv(test_path, required_columns=['text', 'label'])
    
    X_train_text = train_df['text'].fillna('')
    X_test_text = test_df['text'].fillna('')
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    return X_train_text, X_test_text, y_train, y_test


def _build_lr_model(C: float = None) -> LogisticRegression:
    """Build a config-aligned LR model for ablation runs."""
    return LogisticRegression(
        C=C if C is not None else cfg.LR.C,
        max_iter=cfg.LR.max_iter,
        solver='liblinear',
        class_weight=cfg.LR.class_weight,
        random_state=cfg.RANDOM_STATE,
        n_jobs=cfg.LR.n_jobs,
    )


def ablation_tfidf_vocab_size(X_train_text, X_test_text, y_train, y_test) -> List[Dict]:
    """
    Ablation: Impact of TF-IDF vocabulary size on LR performance.
    
    Tests vocabulary sizes: 1000, 5000, 10000, 20000, 50000
    """
    log.info("\n" + "="*60)
    log.info("ABLATION 1: TF-IDF Vocabulary Size")
    log.info("="*60)
    
    vocab_sizes = [1000, 5000, 10000, 20000, 50000]
    results = []
    
    for vocab_size in vocab_sizes:
        start = time.time()
        
        vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            ngram_range=cfg.TFIDF.ngram_range,
            sublinear_tf=cfg.TFIDF.sublinear_tf
        )
    
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        model = _build_lr_model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        elapsed = time.time() - start
        
        results.append({
            'vocab_size': vocab_size,
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
            'time_s': round(elapsed, 2)
        })
        
        log.info(f"Vocab={vocab_size:>6}: Acc={acc:.4f}, F1={f1:.4f} ({elapsed:.1f}s)")
    
    return results


def ablation_ngram_range(X_train_text, X_test_text, y_train, y_test) -> List[Dict]:
    """
    Ablation: Impact of n-gram range on LR performance.
    
    Tests: unigrams only, bigrams, trigrams, (1,3)
    """
    log.info("\n" + "="*60)
    log.info("ABLATION 2: N-gram Range")
    log.info("="*60)
    
    ngram_configs = [
        ((1, 1), "Unigrams"),
        ((1, 2), "Uni+Bigrams"),
        ((1, 3), "Uni+Bi+Trigrams"),
        ((2, 2), "Bigrams only"),
        ((2, 3), "Bi+Trigrams"),
    ]
    
    results = []
    
    for ngram_range, label in ngram_configs:
        start = time.time()
        
        vectorizer = TfidfVectorizer(
            max_features=cfg.TFIDF.max_features,
            ngram_range=ngram_range,
            sublinear_tf=cfg.TFIDF.sublinear_tf
        )
        
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        model = _build_lr_model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        elapsed = time.time() - start
        
        results.append({
            'ngram_range': str(ngram_range),
            'label': label,
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
            'time_s': round(elapsed, 2)
        })
        
        log.info(f"  {label:>18} {str(ngram_range)}: Acc={acc:.4f}, F1={f1:.4f}")
    
    return results


def ablation_word_segmentation(X_train_text, X_test_text, y_train, y_test) -> List[Dict]:
    """
    Ablation: Impact of Vietnamese word segmentation.
    
    Compares: with segmentation (processed text) vs without (raw text).
    """
    log.info("\n" + "="*60)
    log.info("ABLATION 3: Word Segmentation Impact")
    log.info("="*60)
    
    results = []
    
    for use_seg, label in [(True, "Có tách từ"), (False, "Không tách từ")]:
        if use_seg:
            X_train_text_seg = X_train_text
        else:
            # Remove underscore connections (undo word segmentation)
            X_train_text_seg = X_train_text.str.replace('_', ' ', regex=False)
        
        if use_seg:
            X_test_text_seg = X_test_text
        else:
            X_test_text_seg = X_test_text.str.replace('_', ' ', regex=False)
        
        vectorizer = TfidfVectorizer(
            max_features=cfg.TFIDF.max_features,
            ngram_range=cfg.TFIDF.ngram_range,
            sublinear_tf=cfg.TFIDF.sublinear_tf
        )
        
        X_train = vectorizer.fit_transform(X_train_text_seg)
        X_test = vectorizer.transform(X_test_text_seg)
        
        model = _build_lr_model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'config': label,
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
        })
        
        log.info(f"  {label}: Acc={acc:.4f}, F1={f1:.4f}")
    
    return results


def ablation_lr_regularization(X_train_text, X_test_text, y_train, y_test) -> List[Dict]:
    """
    Ablation: Impact of regularization strength on LR.
    """
    log.info("\n" + "="*60)
    log.info("ABLATION 4: Regularization Strength (LR)")
    log.info("="*60)
    
    c_values = [0.01, 0.1, 1, 10, 100]
    results = []
    
    vectorizer = TfidfVectorizer(
        max_features=cfg.TFIDF.max_features,
        ngram_range=cfg.TFIDF.ngram_range,
        sublinear_tf=cfg.TFIDF.sublinear_tf
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    for C in c_values:
        model = _build_lr_model(C=C)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'C': C,
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
        })
        
        log.info(f"C={C:>6}: Acc={acc:.4f}, F1={f1:.4f}")
    
    return results


def ablation_sublinear_tf(X_train_text, X_test_text, y_train, y_test) -> List[Dict]:
    """
    Ablation: Impact of sublinear TF scaling.
    """
    log.info("\n" + "="*60)
    log.info("ABLATION 5: Sublinear TF Scaling")
    log.info("="*60)
    
    results = []
    
    for sublinear in [False, True]:
        label = "TF dưới tuyến tính" if sublinear else "TF chuẩn"
        
        vectorizer = TfidfVectorizer(
            max_features=cfg.TFIDF.max_features,
            ngram_range=cfg.TFIDF.ngram_range,
            sublinear_tf=sublinear
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        
        model = _build_lr_model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'config': label,
            'sublinear_tf': sublinear,
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
        })
        
        log.info(f"  {label}: Acc={acc:.4f}, F1={f1:.4f}")
    
    return results


def main():
    """Run all ablation studies."""


    print("="*60)
    print("ABLATION STUDY")
    print("="*60)
    print("Evaluating component contributions to model performance")
    
    # Load data
    X_train_text, X_test_text, y_train, y_test = load_text_data()
    print(f"\nDataset: {len(X_train_text)} train, {len(X_test_text)} test")
    
    all_results = {}
    
    # Run ablation studies
    all_results['vocab_size'] = ablation_tfidf_vocab_size(
        X_train_text, X_test_text, y_train, y_test
    )
    
    all_results['ngram_range'] = ablation_ngram_range(
        X_train_text, X_test_text, y_train, y_test
    )
    
    all_results['word_segmentation'] = ablation_word_segmentation(
        X_train_text, X_test_text, y_train, y_test
    )
    
    all_results['regularization'] = ablation_lr_regularization(
        X_train_text, X_test_text, y_train, y_test
    )
    
    all_results['sublinear_tf'] = ablation_sublinear_tf(
        X_train_text, X_test_text, y_train, y_test
    )
    
    # Save results
    results_dir = cfg.PATHS.tables_dir
    os.makedirs(results_dir, exist_ok=True)
    
    output = {
        'description': 'Ablation study results',
        'timestamp': datetime.now().isoformat(),
        'ablations': all_results
    }
    
    with open(os.path.join(results_dir, 'ablation_study.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    # Create summary table
    summary_rows = []
    
    # Best vocab size
    best_vocab = max(all_results['vocab_size'], key=lambda x: x['f1_macro'])
    summary_rows.append({
        'Component': 'Kích thước từ vựng',
        'Best Config': f"{best_vocab['vocab_size']}",
        'F1-Score': best_vocab['f1_macro']
    })
    
    # Best ngram
    best_ngram = max(all_results['ngram_range'], key=lambda x: x['f1_macro'])
    summary_rows.append({
        'Component': 'Phạm vi N-gram',
        'Best Config': best_ngram['label'],
        'F1-Score': best_ngram['f1_macro']
    })
    
    # Segmentation impact
    seg_results = all_results['word_segmentation']
    for r in seg_results:
        summary_rows.append({
            'Component': 'Tách từ',
            'Best Config': r['config'],
            'F1-Score': r['f1_macro']
        })
    
    # Sublinear TF
    for r in all_results['sublinear_tf']:
        summary_rows.append({
            'Component': 'Co giãn TF',
            'Best Config': r['config'],
            'F1-Score': r['f1_macro']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, 'ablation_summary.csv'), index=False)
    
    # Save LaTeX table
    with open(os.path.join(results_dir, 'ablation_study.tex'), 'w') as f:
        f.write("% Kết quả nghiên cứu khả trừ\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Nghiên cứu khả trừ: Tác động của các thành phần chính}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{llcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Thành phần} & \\textbf{Cấu hình} & \\textbf{Độ chính xác} & \\textbf{F1-Score} \\\\\n")
        f.write("\\midrule\n")
        
        # Vocab size
        f.write("\\multirow{5}{*}{Kích thước từ vựng}")
        for r in all_results['vocab_size']:
            best = " $\\star$" if r == best_vocab else ""
            f.write(f" & {r['vocab_size']:,} & {r['accuracy']:.4f} & {r['f1_macro']:.4f}{best} \\\\\n")
        
        f.write("\\midrule\n")
        
        # N-gram
        f.write("\\multirow{5}{*}{Phạm vi N-gram}")
        for r in all_results['ngram_range']:
            best = " $\\star$" if r == best_ngram else ""
            f.write(f" & {r['label']} & {r['accuracy']:.4f} & {r['f1_macro']:.4f}{best} \\\\\n")
        
        f.write("\\midrule\n")
        
        # Segmentation
        f.write("\\multirow{2}{*}{Tách từ}")
        for r in all_results['word_segmentation']:
            f.write(f" & {r['config']} & {r['accuracy']:.4f} & {r['f1_macro']:.4f} \\\\\n")
        
        f.write("\\midrule\n")
        
        # Sublinear TF
        f.write("\\multirow{2}{*}{Co giãn TF}")
        for r in all_results['sublinear_tf']:
            f.write(f" & {r['config']} & {r['accuracy']:.4f} & {r['f1_macro']:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    
    return all_results


if __name__ == "__main__":
    results = main()
