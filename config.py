"""
Central configuration for FakeNewsDetector.

All hyperparameters, paths, and constants live here.
Import this module instead of scattering magic numbers across scripts.

Usage:
    from config import cfg
    print(cfg.RANDOM_STATE)
    print(cfg.BILSTM.EMBEDDING_DIM)
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


# ─────────────────────────────────────────────────────────────
# Data / split settings
# ─────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    random_state:   int   = 42
    train_ratio:    float = 0.70
    val_ratio:      float = 0.15
    test_ratio:     float = 0.15
    min_word_count: int   = 5       # rows with fewer words are removed as noise


# ─────────────────────────────────────────────────────────────
# TF-IDF feature settings
# ─────────────────────────────────────────────────────────────
@dataclass
class TFIDFConfig:
    max_features: int   = 10_000
    ngram_range:  tuple = (1, 2)
    min_df:       int   = 2
    max_df:       float = 0.95
    sublinear_tf: bool  = True


# ─────────────────────────────────────────────────────────────
# Logistic Regression settings
# ─────────────────────────────────────────────────────────────
@dataclass
class LRConfig:
    C:            float = 1.0
    max_iter:     int   = 1000
    class_weight: str   = "balanced"
    random_state: int   = 42
    n_jobs:       int   = -1
    # GridSearchCV param grid
    param_grid: dict = field(default_factory=lambda: {
        "C":       [0.1, 1, 10],
        "solver":  ["lbfgs", "liblinear"],
        "max_iter":[1000],
    })
    cv_folds: int = 5


# ─────────────────────────────────────────────────────────────
# SVM settings
# ─────────────────────────────────────────────────────────────
@dataclass
class SVMConfig:
    kernel:       str   = "rbf"
    C:            float = 1.0
    gamma:        str   = "scale"
    class_weight: str   = "balanced"
    random_state: int   = 42
    param_grid: dict = field(default_factory=lambda: {
        "C":      [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma":  ["scale"],
    })
    cv_folds: int = 5


# ─────────────────────────────────────────────────────────────
# BiLSTM settings
# ─────────────────────────────────────────────────────────────
@dataclass
class BiLSTMConfig:
    embedding_dim:  int   = 256
    hidden_dim:     int   = 128
    num_layers:     int   = 2
    dropout:        float = 0.3
    learning_rate:  float = 1e-3
    weight_decay:   float = 1e-5
    batch_size:     int   = 32
    epochs:         int   = 20
    patience:       int   = 5      # early stopping patience
    class_weight:   str   = "balanced"


# ─────────────────────────────────────────────────────────────
# PhoBERT settings
# ─────────────────────────────────────────────────────────────
@dataclass
class PhoBERTConfig:
    model_name:                  str   = "vinai/phobert-base"
    num_classes:                 int   = 2
    dropout:                     float = 0.1
    learning_rate:               float = 2e-5
    weight_decay:                float = 0.01
    warmup_ratio:                float = 0.1
    batch_size:                  int   = 16
    epochs:                      int   = 5
    patience:                    int   = 3      # early stopping patience
    gradient_accumulation_steps: int   = 2
    max_seq_len:                 int   = 256
    class_weight:                str   = "balanced"


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


# Singleton — use `from config import cfg`
cfg = Config()
