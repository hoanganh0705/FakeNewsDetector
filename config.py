"""
Central configuration for FakeNewsDetector.

All hyperparameters, paths, and constants live here.
Import this module instead of scattering magic numbers across scripts.

"""

import os
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
# Project root (one level up from this file)
# ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
@dataclass
class Paths:
    # Data
    raw_data:         str = os.path.join(ROOT_DIR, "data", "raw", "raw.csv")
    segmented_data:   str = os.path.join(ROOT_DIR, "data", "processed", "segmented.csv")
    splits_dir:       str = os.path.join(ROOT_DIR, "data", "splits")
    features_dir:     str = os.path.join(ROOT_DIR, "data", "features")

    # Feature sub-dirs
    tfidf_dir:        str = os.path.join(ROOT_DIR, "data", "features", "tfidf")
    embedding_dir:    str = os.path.join(ROOT_DIR, "data", "features", "embedding")
    phobert_dir:      str = os.path.join(ROOT_DIR, "data", "features", "phobert")

    # Experiments
    experiments_dir:  str = os.path.join(ROOT_DIR, "experiments")
    lr_dir:           str = os.path.join(ROOT_DIR, "experiments", "lr")
    svm_dir:          str = os.path.join(ROOT_DIR, "experiments", "svm")
    bilstm_dir:       str = os.path.join(ROOT_DIR, "experiments", "bilstm")
    bert_dir:         str = os.path.join(ROOT_DIR, "experiments", "bert")

    # Results
    results_dir:      str = os.path.join(ROOT_DIR, "results")
    figures_dir:      str = os.path.join(ROOT_DIR, "results", "figures")
    tables_dir:       str = os.path.join(ROOT_DIR, "results", "tables")

    # Paper
    paper_dir:        str = os.path.join(ROOT_DIR, "paper")
    paper_figures_dir: str = os.path.join(ROOT_DIR, "paper", "figures")
    paper_tables_dir: str = os.path.join(ROOT_DIR, "paper", "tables")


# ─────────────────────────────────────────────────────────────
# Data / split settings
# ─────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    train_ratio:    float = 0.70
    val_ratio:      float = 0.15
    test_ratio:     float = 0.15
    min_word_count: int   = 10       # rows with fewer words are removed as noise


# ─────────────────────────────────────────────────────────────
# TF-IDF feature settings
# ─────────────────────────────────────────────────────────────
# Dataset note: median=25 words, P75=62. Many short texts → trigrams
# help capture more Vietnamese compound expressions even in short docs.
@dataclass
class TFIDFConfig:
    max_features: int   = 40000          # increased from 50K to fit trigrams
    ngram_range:  tuple = (1, 2)          # unigrams + bigrams
    min_df:       int   = 5               # lowered from 3 — 15K docs is enough
    max_df:       float = 0.95
    sublinear_tf: bool  = True


# ─────────────────────────────────────────────────────────────
# Logistic Regression settings
# ─────────────────────────────────────────────────────────────
# Analysis: best was C=10 saga (val_f1=0.8454, test_f1=0.8329) via 5-fold GridSearchCV.
@dataclass
class LRConfig:
    C:            float = 1.0
    max_iter:     int   = 3000
    class_weight: str   = "balanced"
    n_jobs:       int   = -1
    # GridSearchCV param grid — expanded with saga + elasticnet
    param_grid: dict = field(default_factory=lambda: {
        "C":       [0.5, 1, 5, 10, 20, 50],    # finer around best=10
        "solver":  ["liblinear", "saga"],       # saga enables elasticnet
        "penalty": ["l2"],                      # l2 works for both solvers
        "max_iter":[3000],
    })
    cv_folds: int = 5


# ─────────────────────────────────────────────────────────────
# SVM settings
# ─────────────────────────────────────────────────────────────
# Analysis: best was LinearSVC C=0.5 (test_acc=0.8434, test_f1=0.8410) via GridSearchCV.
@dataclass
class SVMConfig:
    kernel:       str   = "rbf"
    C:            float = 1.0
    gamma:        str   = "scale"
    class_weight: str   = "balanced"
    # When use_linear=True the trainer will use LinearSVC + calibration.
    use_linear:   bool  = True
    # Param grid for non-linear SVC (used when use_linear=False)
    param_grid: dict = field(default_factory=lambda: {
        "C":      [0.1, 1, 10],
        "kernel": ["rbf"],
        "gamma":  ["scale"],
    })
    # Param grid for LinearSVC (finer grid centered on C=1)
    linear_param_grid: dict = field(default_factory=lambda: {
        "C": [0.1, 0.5, 1, 2, 5, 10]
    })
    cv_folds: int = 5


# ─────────────────────────────────────────────────────────────
# BiLSTM settings
# ─────────────────────────────────────────────────────────────
# Final config: hidden_dim=128, 1 layer, dropout=0.3 — balanced between overfitting (384) and underfitting (frozen 128).
@dataclass
class BiLSTMConfig:
    embedding_dim:   int   = 300           # matches FastText cc.vi.300
    hidden_dim:      int   = 128           # middle ground between 384 (overfit) and 128 (underfit)
    num_layers:      int   = 1
    dropout:         float = 0.3         # aggressive dropout for 10K data
    learning_rate:   float = 1e-3         # slower convergence
    weight_decay:    float = 1e-4        # strong L2 regularisation
    batch_size:      int   = 64
    epochs:          int   = 20
    patience:        int   = 5             # stop sooner when plateaued
    max_seq_length:  int   = 128           # median is 25, P90 is 247
    class_weight:    str   = None
    label_smoothing: float = 0.0          # reverted — helps fight overfitting
    freeze_embeddings: bool = False        # unfreezing — frozen caused underfitting
    # fasttext_path: set to .bin file path to use pretrained Vietnamese FastText
    # Download: https://fasttext.cc/docs/en/crawl-vectors.html (cc.vi.300.bin)
    fasttext_path:   str   = os.path.join(ROOT_DIR, "data", "fasttext", "cc.vi.300.bin")


# ─────────────────────────────────────────────────────────────
# PhoBERT settings
# ─────────────────────────────────────────────────────────────
# Dataset note: only ~11K train samples with a 355M-param model →
# strong regularisation (dropout, label smoothing, layer LR decay) is key.
# P90 ≈ 321 subword tokens, but PhoBERT max is 256 — unavoidable truncation.
@dataclass
class PhoBERTConfig:
    model_name:                  str   = "vinai/phobert-base"  # upgraded from phobert-base
    num_classes:                 int   = 2
    dropout:                     float = 0.1
    learning_rate:               float = 3e-5
    weight_decay:                float = 0.01
    warmup_ratio:                float = 0.1

    batch_size:                  int   = 16      # reduced for phobert-large VRAM
    epochs:                      int   = 8      # BERT paper recommends 2-4; 6 is safe upper bound
    patience:                    int   = 2      # stop fast — overfit happens in 2-3 epochs

    gradient_accumulation_steps: int   = 4      # effective batch ~64 (was 2)
    max_seq_len:                 int   = 256    # phobert-large max_position_embeddings=258

    class_weight:                str   = None
    label_smoothing:             float = 0.0   # label-smoothed cross-entropy
    layer_lr_decay:              float = 0.95   # per-layer LR decay factor


# ─────────────────────────────────────────────────────────────
# Analysis / Reporting settings
# ─────────────────────────────────────────────────────────────
@dataclass
class AnalysisConfig:
    bootstrap_iterations: int = 10000
    significance_level: float = 0.05
    # Add more reporting-related defaults here if needed

# ─────────────────────────────────────────────────────────────
# Root config object — import this everywhere
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    RANDOM_STATE: int = 42

    PATHS:   Paths      = field(default_factory=Paths)
    DATA:    DataConfig = field(default_factory=DataConfig)
    TFIDF:   TFIDFConfig = field(default_factory=TFIDFConfig)
    LR:      LRConfig    = field(default_factory=LRConfig)
    SVM:     SVMConfig   = field(default_factory=SVMConfig)
    BILSTM:  BiLSTMConfig  = field(default_factory=BiLSTMConfig)
    PHOBERT: PhoBERTConfig = field(default_factory=PhoBERTConfig)
    ANALYSIS: AnalysisConfig = field(default_factory=AnalysisConfig)


# Singleton — use `from config import cfg`
cfg = Config()
