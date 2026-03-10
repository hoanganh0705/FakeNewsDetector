# FakeNewsDetector — Complete Study Guide

> Written for someone who vibe-coded the project and needs to understand **every line for a presentation**. This covers what the code does, why each design decision exists, and the actual results — all verified against the real source files and `experiments/*/metrics.json`.

---

## Table of Contents

1. [Big Picture — What This Project Does](#1-big-picture)
2. [Project Structure Map](#2-project-structure-map)
3. [Full Pipeline Walkthrough](#3-full-pipeline-walkthrough)
   - [Step 0: Configuration (config.py)](#step-0-configuration)
   - [Step 1: Word Segmentation](#step-1-word-segmentation)
   - [Step 2: Data Splitting](#step-2-data-splitting)
   - [Step 3: Feature Extraction](#step-3-feature-extraction)
   - [Step 4: Model Training](#step-4-model-training)
   - [Step 5: Evaluation](#step-5-evaluation)
   - [Step 6: Error Analysis](#step-6-error-analysis)
   - [Step 7: Statistical Tests](#step-7-statistical-tests)
   - [Step 8: Cross-Validation](#step-8-cross-validation)
   - [Step 9: Ablation Study](#step-9-ablation-study)
   - [Step 10: Explainability](#step-10-explainability)
   - [Step 11–12: Paper Figures & Tables](#step-1112-paper-figures--tables)
4. [Concepts You Need to Understand](#4-concepts-you-need-to-understand)
5. [Model Architecture Deep-Dive](#5-model-architecture-deep-dive)
6. [Actual Results (from metrics.json)](#6-actual-results)
7. [Quick Reference](#7-quick-reference)

---

## 1. Big Picture

### What This Is

A **Vietnamese fake news detection** research project. Given a Vietnamese news article, classify it as **Thật / Real (label=0)** or **Giả / Fake (label=1)**.

### Why It's a Research Paper (Not Just an App)

The project compares **4 different ML approaches** on the same dataset to answer: *"How much better are modern transformers over classical ML for Vietnamese NLP?"*

```
Raw Vietnamese Text (data/raw/raw.csv)
           │
           ▼ Step 1: Vietnamese word segmentation (VnCoreNLP RDRSegmenter)
   Segmented Text (data/processed/segmented.csv)
           │
           ▼ Step 2: Clean + stratified split into train/val/test
   data/splits/{train, val, test}.csv
           │
    ┌──────┴──────────────────────┐
    │                             │
    ▼ Step 3a                     ▼ Step 3b/3c
TF-IDF vectors             Word IDs / BERT tokens
(data/features/tfidf/)     (data/features/embedding/, phobert/)
    │                             │
    ├── Logistic Regression       ├── BiLSTM
    └── LinearSVC                 └── PhoBERT fine-tuning
           │                             │
           └──────────┬──────────────────┘
                      ▼ Steps 5–12
                 Evaluation, Analysis, Paper
```

### The Answer

PhoBERT (Vietnamese BERT pretrained on 20GB of Vietnamese text) achieves **90.1% accuracy** vs **82.5–84.3%** for classic models — a meaningful improvement that justifies the research.

---

## 2. Project Structure Map

```
FakeNewsDetector/
│
├── config.py                    ← ALL hyperparameters live here (single source of truth)
│
├── data/
│   ├── raw/raw.csv              ← Input: 16,946 articles (id, text, date, label)
│   ├── processed/segmented.csv  ← After word segmentation: 15,789 articles
│   ├── splits/                  ← Train(9,770) / Val(2,094) / Test(2,094)
│   └── features/                ← Extracted numeric features (ready for models)
│       ├── tfidf/               ← Sparse matrix for LR & SVM
│       ├── embedding/           ← Integer sequences for BiLSTM
│       └── phobert/             ← Token IDs + attention masks for PhoBERT
│
├── src/
│   ├── preprocessing/           ← Text cleaning, segmentation, splitting
│   ├── features/                ← Convert text → numbers
│   ├── models/                  ← Neural network architecture definitions
│   ├── training/                ← Training loops (hyperparameter tuning, saving)
│   ├── evaluation/              ← Metrics, error analysis, ablation, CV
│   ├── analysis/                ← Stats tests, explainability, paper outputs
│   └── utils/                   ← Logger, helper tools
│
├── experiments/                 ← Saved model weights + metrics.json (actual results)
│   ├── lr/                      ← lr_model.pkl, metrics.json
│   ├── svm/                     ← svm_model.pkl, metrics.json
│   ├── bilstm/                  ← bilstm_model.pt, metrics.json
│   └── bert/                    ← phobert_model.pt, metrics.json
│
├── results/
│   ├── figures/                 ← PNG/PDF plots (confusion matrices, ROC curves, etc.)
│   └── tables/                  ← CSV + JSON + LaTeX results
│
└── paper/
    ├── figures/                 ← Publication-quality PDF+PNG figures
    └── tables/                  ← LaTeX .tex files ready for \input{}
```

---

## 3. Full Pipeline Walkthrough

### Step 0: Configuration

**File:** [config.py](config.py)

The **central brain** of the project. Every hyperparameter, path, and constant is defined once here using Python `@dataclass`. Every other script imports:

```python
from config import cfg
```

**Why this matters for a research paper:** reproducibility. If numbers are scattered everywhere, you cannot reliably re-run the same experiment. `config.py` is the single place you change when tuning.

#### All Key Values (from the actual file)

```python
cfg.RANDOM_STATE = 42           # used in all random calls → reproducible splits

# Data
cfg.DATA.train_ratio = 0.70
cfg.DATA.val_ratio   = 0.15
cfg.DATA.test_ratio  = 0.15
cfg.DATA.min_word_count = 10    # articles with <10 words are dropped as noise

# TF-IDF
cfg.TFIDF.max_features = 40000  # vocabulary cap (40K most informative words/bigrams)
cfg.TFIDF.ngram_range  = (1, 2) # unigrams + bigrams
cfg.TFIDF.min_df       = 5      # word must appear in ≥5 documents
cfg.TFIDF.max_df       = 0.95   # word must appear in <95% of documents
cfg.TFIDF.sublinear_tf = True   # use log(1+count) instead of raw count

# Logistic Regression
cfg.LR.C         = 1.0          # default; GridSearchCV overrides to C=10
cfg.LR.max_iter  = 3000
cfg.LR.param_grid = {"C": [0.5,1,5,10,20,50], "solver": ["liblinear","saga"], ...}

# SVM
cfg.SVM.use_linear = True       # use LinearSVC + CalibratedClassifierCV (fast)
cfg.SVM.linear_param_grid = {"C": [0.1, 0.5, 1, 2, 5, 10]}

# BiLSTM
cfg.BILSTM.embedding_dim   = 300    # matches FastText cc.vi.300 pretrained vectors
cfg.BILSTM.hidden_dim      = 128
cfg.BILSTM.num_layers      = 1
cfg.BILSTM.dropout         = 0.3
cfg.BILSTM.learning_rate   = 1e-3
cfg.BILSTM.weight_decay    = 1e-4
cfg.BILSTM.batch_size      = 64
cfg.BILSTM.epochs          = 20     # max; early stopping fires first
cfg.BILSTM.patience        = 5
cfg.BILSTM.max_seq_length  = 128
cfg.BILSTM.fasttext_path   = "data/fasttext/cc.vi.300.bin"

# PhoBERT
cfg.PHOBERT.model_name                  = "vinai/phobert-base"
cfg.PHOBERT.dropout                     = 0.1
cfg.PHOBERT.learning_rate               = 3e-5
cfg.PHOBERT.weight_decay                = 0.01
cfg.PHOBERT.warmup_ratio                = 0.1
cfg.PHOBERT.batch_size                  = 16
cfg.PHOBERT.epochs                      = 8
cfg.PHOBERT.patience                    = 2
cfg.PHOBERT.gradient_accumulation_steps = 4   # effective batch = 64
cfg.PHOBERT.max_seq_len                 = 256
cfg.PHOBERT.label_smoothing             = 0.0
cfg.PHOBERT.layer_lr_decay              = 0.95 # per-layer LR decay

# Analysis
cfg.ANALYSIS.bootstrap_iterations = 10000
cfg.ANALYSIS.significance_level   = 0.05
```

#### `@dataclass` Explained (for beginners)

```python
@dataclass
class TFIDFConfig:
    max_features: int   = 40000
    ngram_range:  tuple = (1, 2)
```

`@dataclass` is a Python decorator that auto-generates `__init__`, `__repr__`, etc. for a class whose purpose is just to hold data. Cleaner than a plain `dict` because you get type hints and IDE auto-complete.

---

### Step 1: Word Segmentation

**File:** [src/preprocessing/word_segmentation.py](src/preprocessing/word_segmentation.py)
**Input:** `data/raw/raw.csv` (16,946 rows) → **Output:** `data/processed/segmented.csv` (15,789 rows)

```bash
python src/preprocessing/word_segmentation.py
```

#### Why Vietnamese Is Different

In English, words are separated by spaces. In Vietnamese, many concepts are **compound words written with spaces between their syllables**:

| Without segmentation | With segmentation |
|---|---|
| `thành phố Hồ Chí Minh` → 5 tokens | `thành_phố Hồ_Chí_Minh` → 2 tokens |
| `Việt Nam` → 2 tokens | `Việt_Nam` → 1 token |
| `học sinh` → 2 tokens | `học_sinh` → 1 token |

The segmenter joins compound syllables with underscores so the model sees correct semantic units.

#### Which Segmenter?

The code uses **`py_vncorenlp` (VnCoreNLP RDRSegmenter)** — the **exact same tokenizer PhoBERT was trained with**. This is critical: if you use a different segmenter, the token shapes won't match PhoBERT's vocabulary and performance drops.

```python
# Priority 1: py_vncorenlp (VinAI RDRSegmenter — matches PhoBERT training)
# Priority 2: underthesea  (pure-Python fallback when Java unavailable)
# The fallback is automatic; the script never hard-fails.
```

On first run, `py_vncorenlp` downloads model weights to `.vncorenlp/` and caches them — no manual download needed.

#### What to Learn

- Vietnamese NLP requires word segmentation before anything else
- Tokenization consistency matters: use the same tokenizer for ALL models in a comparison
- Compound words = meaningful semantic units that happen to use spaces in Vietnamese

---

### Step 2: Data Splitting

**File:** [src/preprocessing/split_data.py](src/preprocessing/split_data.py)
**Input:** `data/processed/segmented.csv` → **Output:** `data/splits/{train,val,test}.csv`

```bash
python src/preprocessing/split_data.py
```

#### What It Does (3 things)

1. **Cleans data** — removes duplicates, empty texts, and articles with fewer than `cfg.DATA.min_word_count = 10` words
2. **Stratified split** — 70% train / 15% val / 15% test, preserving the Real/Fake ratio in each split
3. **Data leakage check** — verifies zero overlap between all three sets

#### Actual Dataset Numbers (verified from CSV `wc -l`)

| Stage | Count |
|---|---|
| Raw articles | 16,946 |
| After segmentation | 15,789 |
| After cleaning (≥10 words) | ~13,958 |
| **Train** | **9,770** |
| **Val** | **2,094** |
| **Test** | **2,094** |
| Test Real (Thật) | 1,165 (55.6%) |
| Test Fake (Giả) | 929 (44.4%) |

#### Why Stratified Split?

The dataset has more Real news than Fake news (~55.6% Real / 44.4% Fake). A naive random split might give you 65% Real in test but only 50% Real in train — your reported test metric would not reflect real-world performance. **Stratified** means each split preserves the original class ratio.

#### Train / Val / Test Roles

| Set | Purpose | When used |
|---|---|---|
| **Train** | Model learns from this — parameters updated | During training |
| **Val** (validation) | Tune hyperparameters — NOT in final report | After each epoch; in GridSearchCV |
| **Test** | **Final honest evaluation** — touch ONCE at the end | One time only |

**Data leakage = catastrophic bug.** If any test example also appears in training, you are measuring memorization, not generalization. The split script explicitly checks for this.

#### What to Learn

- Stratified sampling
- Train / val / test purpose and isolation
- Data leakage and why it invalidates results

---

### Step 3: Feature Extraction

**Files:** [src/features/](src/features/)
**Input:** `data/splits/*.csv` → **Output:** `data/features/*/`

```bash
python src/features/extract_all_features.py
```

Computers cannot process raw text. This step converts text into numbers. Three different representations are produced for the four models.

#### 3a. TF-IDF (for Logistic Regression & LinearSVC)

**File:** [src/features/tfidf_features.py](src/features/tfidf_features.py)

Converts each article into a **40,000-dimensional sparse vector**. Each dimension represents a word or bigram. The value is:

$$\text{TF-IDF}(t, d) = \underbrace{\log(1 + \text{count}(t, d))}_{\text{sublinear TF}} \times \underbrace{\log \frac{N}{\text{docs containing } t}}_{\text{IDF}}$$

- **High TF-IDF:** rare word that appears often in *this* document → probably important
- **Low TF-IDF:** common word like "và", "của", "là" → not discriminative

**Key settings (from config.py):**

| Setting | Value | Meaning |
|---|---|---|
| `max_features` | 40,000 | Keep only top 40K most informative words/bigrams |
| `ngram_range` | (1, 2) | Single words AND 2-word phrases |
| `min_df` | 5 | Word must appear in ≥5 documents (removes typos, rare noise) |
| `max_df` | 0.95 | Word must appear in <95% of docs (removes stop words) |
| `sublinear_tf` | True | `log(1+count)` dampens very high term frequencies |

Saved as a **sparse matrix** (`.pkl`). "Sparse" = most values are 0, only non-zero entries stored — memory efficient for 40K dimensions.

#### 3b. Word Embeddings (for BiLSTM)

**File:** [src/features/embedding_features.py](src/features/embedding_features.py)

Builds a vocabulary (~23,000 words) and converts each article to a **sequence of integer IDs**:

```
"Thủ_tướng họp báo hôm_nay" → [142, 891, 3, 0, 0, ..., 0]
                                            ↑ padding to max_seq_length=128
```

Special tokens:
- `<PAD>` (index 0) — pads short sequences to fixed length 128
- `<UNK>` (index 1) — represents words not seen during training

The BiLSTM loads **FastText pretrained embeddings** (`cc.vi.300.bin` — 300-dimensional vectors trained on Vietnamese Wikipedia + Common Crawl). These give the mapping: `word_id → 300-dim float vector`.

**Why pretrained embeddings?** Training embeddings from scratch on ~10K examples is poor. FastText was trained on billions of Vietnamese words; its vectors encode semantic similarity (e.g., "tổng_thống" and "thủ_tướng" have similar vectors because they appear in similar contexts).

Saved as: `data/features/embedding/embedding_features.pkl`

#### 3c. PhoBERT Tokenization (for PhoBERT)

**File:** [src/features/phobert_features.py](src/features/phobert_features.py)

Uses the PhoBERT **Byte-Pair Encoding (BPE) tokenizer** (64,000 subword vocabulary) to produce:
- `input_ids` — integer ID for each subword token
- `attention_mask` — 1 for real tokens, 0 for padding

**Max length:** 256 tokens. Articles longer than 256 subword tokens are **truncated** (some content is lost). Config notes that P90 ≈ 321 subword tokens — roughly 10% of articles are truncated.

**BPE explained:** splits rare words into subword pieces:
- `"bầu_cử"` → `["bầu", "##cử"]` (compound stays together)
- `"COVID19hoax"` → `["CO", "##VI", "##D", "##19", "##ho", "##ax"]` (unknown → pieces)

The model never sees a fully unknown token — it always has *something* to work with.

Tokenizer + model are cached locally at `data/features/phobert_tokenizer_cache/` and `data/features/phobert_model_cache/` after the first download.

Saved as: `data/features/phobert/phobert_features.pkl`

#### What to Learn

- TF-IDF intuition: rare-but-present words are most informative
- Sparse vs dense matrix
- Word embeddings vs one-hot encoding (embeddings encode similarity; one-hot doesn't)
- BPE tokenization and subword units
- `<PAD>`, `<UNK>` special tokens
- Transfer learning via pretrained embeddings

---

### Step 4: Model Training

**Files:** [src/training/](src/training/) and [src/models/](src/models/)

```bash
# All at once:
python src/training/train_all.py

# Or individually:
.venv/bin/python src/training/train_lr.py
.venv/bin/python src/training/train_svm.py
.venv/bin/python src/training/train_bilstm.py
.venv/bin/python src/training/train_phobert.py
```

#### 4a. Logistic Regression

**File:** [src/training/train_lr.py](src/training/train_lr.py)

Takes TF-IDF vectors (40K dimensions) as input. Learns the best **linear decision boundary** — a hyperplane in 40,000-dimensional space separating Real from Fake.

Prediction:

$$P(\text{Fake}) = \sigma\!\left(\sum_{i=1}^{40000} w_i \cdot \text{tfidf}_i + b\right), \quad \sigma(x) = \frac{1}{1+e^{-x}}$$

**GridSearchCV** tries all combinations by 5-fold cross-validation on the training set:

```python
param_grid = {
    "C":       [0.5, 1, 5, 10, 20, 50],
    "solver":  ["liblinear", "saga"],
    "penalty": ["l2"],
    "max_iter": [3000],
}
```

**Best found (from experiments/lr/metrics.json):** `C=10, solver=saga, penalty=l2`

**What `C` does:** regularization strength.
- Small C → model must be simple (weights stay small) → may underfit
- Large C → model can use large weights → may overfit
- C=10 means "allow moderate complexity; trust the training data"

**`solver=saga`** is a stochastic gradient algorithm that scales well to large datasets and supports multiple penalty types.

**`class_weight="balanced"`** automatically upweights the minority class (Fake). Without this, a lazy model can get decent accuracy by predicting "Real" most of the time.

**Actual test results:** Accuracy=83.5%, F1 (macro)=83.3%, AUC=91.9%

#### 4b. SVM (LinearSVC + Calibration)

**File:** [src/training/train_svm.py](src/training/train_svm.py)

⚠️ **Important correction from naive understanding:** Despite `cfg.SVM.kernel = "rbf"` existing in config, the code uses `cfg.SVM.use_linear = True` which selects `LinearSVC` — a highly optimized **linear** support vector classifier.

**Why LinearSVC for text?** With 40,000 features, TF-IDF vectors are already in a high-dimensional space where classes are nearly linearly separable. The RBF kernel's non-linear mapping is not needed — and would be far slower (`O(n²)` vs `O(n)` for linear).

```python
# LinearSVC does not produce probability estimates.
# CalibratedClassifierCV wraps it to enable predict_proba() (needed for ROC-AUC).
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
svm = CalibratedClassifierCV(LinearSVC(C=best_C, class_weight="balanced"))
```

**GridSearchCV grid:** `C ∈ {0.1, 0.5, 1, 2, 5, 10}`
**Best found (from experiments/svm/metrics.json):** `C=0.5`

**SVM intuition:** finds the hyperplane that **maximizes the margin** (gap) between the two classes. Points on the boundary are the "support vectors" — the model depends only on these boundary cases.

**Actual test results:** Accuracy=84.3%, F1 (macro)=84.1%, AUC=91.9%

**Why LinearSVC slightly outperforms LR?** LinearSVC optimizes **hinge loss** (maximize margin), which tends to be more robust than LR's logistic loss for well-separated text classes.

#### 4c. BiLSTM (Bidirectional LSTM with Soft Attention)

**Files:** [src/models/bilstm_model.py](src/models/bilstm_model.py), [src/training/train_bilstm.py](src/training/train_bilstm.py)

Architecture (from `bilstm_model.py`):

```
Input: integer sequence [w1, w2, ..., w128] (padded to max_seq_length=128)
    ↓
Embedding layer (vocab_size × 300)
    │  Each integer → 300-dim pretrained FastText vector
    ↓
BiLSTM (hidden_dim=128, num_layers=1, bidirectional=True)
    │  Forward LSTM:  reads left→right  → 128-dim hidden per timestep
    │  Backward LSTM: reads right→left → 128-dim hidden per timestep
    │  Output per timestep: 128 + 128 = 256 dimensions
    ↓
Soft Attention
    │  Learned weights α_t for each timestep t
    │  context = Σ_t (α_t × hidden_t)   (weighted average over time)
    │  "Which words matter most for this classification?"
    ↓
Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→2)
    ↓
Softmax → [P(Thật), P(Giả)]
```

**Why Bidirectional?** A standard LSTM reads left-to-right only. For "Ông Biden đã ký [MASK] với Trung Quốc", the forward LSTM doesn't know what comes after "ký". The backward LSTM reads right-to-left and does know. Combining both gives full context for every word.

**Why Attention?** Not all words matter equally. Attention assigns weight α_t to each token. Words like "giả_mạo" (forgery) or "bịa_đặt" (fabricated) get high weights; filler words like "và" (and) get near-zero weights. This makes the model's decision interpretable: you can visualize which words it "paid attention to".

**Training details (config.py → confirmed in experiments/bilstm/metrics.json):**

| Setting | Value | Reasoning |
|---|---|---|
| `embedding_dim` | 300 | Matches FastText cc.vi.300 |
| `hidden_dim` | 128 | **Development history below** |
| `num_layers` | 1 | Adding layers caused overfitting on small data |
| `dropout` | 0.3 | Aggressive regularization for ~10K training examples |
| `learning_rate` | 1e-3 | Standard Adam LR for LSTM |
| `weight_decay` | 1e-4 | L2 regularization on weights |
| `batch_size` | 64 | |
| `max_seq_length` | 128 | Median = 25 words; P90 = 247 (some truncation OK) |
| Optimizer | Adam | |
| Loss | CrossEntropyLoss | class-weighted for imbalance |

**The development history (comment in config.py — this shows real iterative research):**
```
Round 1: hidden_dim=384, unfrozen → test_f1=0.8368, severe overfit (99.5% train acc)
Round 2: hidden_dim=128, frozen embeddings → test_f1=0.7907, underfit (84% train acc)
Round 3: hidden_dim=128, unfrozen embeddings, strong dropout/L2 → current setup ✓
```

**Actual test results:** Accuracy=82.5%, F1 (macro)=82.3%, AUC=90.5% (trained for 8 epochs, early stopping patience=5)

#### 4d. PhoBERT (Fine-tuned Transformer)

**Files:** [src/models/phobert_model.py](src/models/phobert_model.py), [src/training/train_phobert.py](src/training/train_phobert.py)

Architecture (from `phobert_model.py`):

```
Input: [CLS] + subword_tokens + [SEP]   (max 256 tokens)
    ↓
PhoBERT encoder (vinai/phobert-base)
    │  12 transformer layers
    │  768 hidden dimensions per token
    │  Each layer: Multi-Head Self-Attention + Feed-Forward + LayerNorm
    │  Self-attention: EVERY token attends to EVERY other token simultaneously
    │  (unlike LSTM which processes tokens sequentially)
    ↓
[CLS] token representation (768-dim)
    │  The special [CLS] token at position 0
    │  After 12 layers of global self-attention, it encodes the whole article
    ↓
Dropout(0.1) → Linear(768→384) → ReLU → Dropout(0.1) → Linear(384→2)
    ↓
Softmax → [P(Thật), P(Giả)]
```

**Why PhoBERT is so much better:**

PhoBERT was pretrained by VinAI on **20GB of Vietnamese text** (news + Wikipedia) with Masked Language Modeling (MLM): randomly mask 15% of tokens, predict them. After pretraining, the model already understands:
- Vietnamese grammar and syntax
- Named entities (politicians, organizations, locations)
- Semantic similarity between words and phrases
- Long-range dependencies in sentences

When we **fine-tune** it (adjust weights slightly for our fake news task), we get deep Vietnamese language understanding for free.

**Training details (config.py → confirmed in experiments/bert/metrics.json):**

| Setting | Value | Reasoning |
|---|---|---|
| `model_name` | `vinai/phobert-base` | Pretrained on 20GB Vietnamese text |
| `learning_rate` | 3e-5 | **Very low** — large LR = catastrophic forgetting of pretraining |
| `batch_size` | 16 | GPU memory constraint |
| `gradient_accumulation_steps` | 4 | Effective batch = 16×4 = **64** |
| `warmup_ratio` | 0.1 | First 10% of steps: LR ramps up from 0→3e-5 |
| `weight_decay` | 0.01 | L2 regularization via AdamW |
| `layer_lr_decay` | 0.95 | Layer 12: LR=3e-5; Layer 1: LR=3e-5×0.95¹¹≈1.67e-5 |
| `max_seq_len` | 256 | PhoBERT max is 258 subword tokens |
| `epochs` | 8 | Early stopping patience=2 |
| Optimizer | AdamW | Like Adam but with proper weight decay |

**Gradient accumulation:** with `batch_size=16` and `accumulation_steps=4`, the optimizer only steps every 4 forward passes. This simulates a batch of 64 without needing GPU memory for 64 examples at once.

**Layer LR decay:** The first BERT layers learn fundamental Vietnamese language structure. The last layers learn task-specific patterns. We update top layers faster (full LR=3e-5) and bottom layers slower (×0.95 per layer), protecting earlier pretrained knowledge.

**Model loading (from `phobert_model.py`):** checks `data/features/phobert_model_cache/` first, downloads from HuggingFace Hub if not present, then saves locally — no internet needed after first run.

**Actual test results:** Accuracy=90.1%, F1 (macro)=89.9%, AUC=95.0% (trained for 8 epochs)

---

### Step 5: Evaluation

**File:** [src/evaluation/evaluate_all.py](src/evaluation/evaluate_all.py)
**Input:** trained models + test data → **Output:** `results/`

```bash
.venv/bin/python src/evaluation/evaluate_all.py
```

Runs each model on the held-out test set and generates:

| Output | Description |
|---|---|
| `results/tables/model_comparison.csv` | Side-by-side metrics for all 4 models |
| `results/tables/per_class_metrics.csv` | Real vs Fake breakdown per model |
| `results/figures/confusion_matrices.png` | 2×2 matrices for all 4 models |
| `results/figures/roc_curves.png` | ROC curves overlaid |
| `results/figures/training_history.png` | BiLSTM/PhoBERT loss + accuracy over epochs |
| `experiments/*/predictions.pkl` | Raw predictions (needed for statistical tests) |

#### All Metrics Explained

```
                        Predicted
                  Real (0)    Fake (1)
Actual  Real (0)    TP          FP    ← FP: flagged as fake but is actually real
        Fake (1)    FN          TN    ← FN: missed a fake (predicted as real)
```

| Metric | Formula | What it means |
|---|---|---|
| **Accuracy** | (TP+TN)/(all) | Overall % correct |
| **Precision** | TP/(TP+FP) | Of all we flagged as fake, how many were really fake? |
| **Recall** | TP/(TP+FN) | Of all actual fakes, how many did we catch? |
| **F1 (macro)** | harmonic mean of P&R, averaged equally across classes | Balances precision and recall for both classes |
| **ROC-AUC** | area under ROC curve | 1.0 = perfect, 0.5 = random guessing |

**Which error is worse in fake news?**
- FP (marking real news as fake) = potential censorship
- FN (missing fake news) = misinformation spreads

Both matter equally, so we use **macro F1** as the primary metric.

**Macro vs Weighted:**
- **Macro avg:** Real F1 + Fake F1 / 2 — treats both classes equally
- **Weighted avg:** weights by class size — favors the majority (Real) class
- We use **macro** because both Real and Fake classification matter equally

---

### Step 6: Error Analysis

**File:** [src/evaluation/error_analysis.py](src/evaluation/error_analysis.py)

```bash
.venv/bin/python src/evaluation/error_analysis.py
```

Understand *where* and *why* models fail — not just *how often*.

**Questions answered:**
- Which examples did **all 4 models** get wrong? → `results/tables/hard_examples.csv`
- Are errors concentrated on short/medium/long articles?
- How often is the model **confidently wrong** (high-confidence error = dangerous)?

**High-confidence wrong prediction:** the model outputs P(Fake)=0.94, but the true label is Real. This is more dangerous than a low-confidence error (P=0.52) because downstream systems that threshold on confidence would fully trust the wrong answer.

**Error Taxonomy categorizes by:**
- Text length: Short (<50 words) / Medium / Long (>200 words)
- Confidence level: High-confidence wrong vs Low-confidence wrong

**Key concepts:**

| Term | Meaning |
|---|---|
| False Positive (FP) | Predicted Fake, actually Real (over-censorship) |
| False Negative (FN) | Predicted Real, actually Fake (missed misinformation) |
| High-confidence error | Model is very wrong AND very sure |
| Hard examples | All 4 models got them wrong → irreducible difficulty |

---

### Step 7: Statistical Significance Tests

**File:** [src/analysis/statistical_tests.py](src/analysis/statistical_tests.py)

```bash
.venv/bin/python src/analysis/statistical_tests.py
```

#### Why This Step Exists

If PhoBERT gets 89.9% F1 and LR gets 83.3% F1, could that difference be due to the specific 2,094-article test set we happened to draw? Statistical tests answer this.

#### McNemar's Test

Compares two classifiers **sample-by-sample** — only looks at examples where one model is right and the other is wrong:

```
              Model B right    Model B wrong
Model A right:    (ignore)          b
Model A wrong:       c           (ignore)
```

**Test statistic:** (|b − c| − 1)² / (b + c) ~ χ²(1)  
**Null hypothesis H₀:** both classifiers have equal error rates (b ≈ c)  
**p-value:** probability of observing this b/c ratio if H₀ is true  
**Decision:** p < 0.05 → reject H₀ → difference is statistically significant

**Holm-Bonferroni correction:** 6 pairwise tests run (LR vs SVM, LR vs BiLSTM, LR vs PhoBERT, SVM vs BiLSTM, SVM vs PhoBERT, BiLSTM vs PhoBERT). Running 6 tests at α=0.05 expects 0.3 false positives. Holm-Bonferroni sorts p-values and applies increasingly strict thresholds to control the family-wise error rate.

#### Bootstrap Confidence Intervals

Resample the test set **10,000 times** with replacement, compute metric each time → get a distribution → take 2.5th and 97.5th percentiles as 95% CI.

```python
cfg.ANALYSIS.bootstrap_iterations = 10000
```

Non-overlapping CIs between two models confirm they are genuinely different.

#### Cohen's d (Effect Size)

Tells you HOW BIG the difference is, not just whether it exists:

| d value | Interpretation |
|---|---|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| > 0.8 | Large |

Two models can have p=0.0001 (very significant) but d=0.03 (practically negligible on this dataset).

#### What to Learn

- Hypothesis testing, null hypothesis, p-value
- Type I error (false positive, α=0.05) vs Type II error (false negative, β)
- Multiple comparisons problem: why you can't run many tests at α=0.05 each
- Holm-Bonferroni correction
- Bootstrap resampling (non-parametric CI estimation)
- Statistical significance ≠ practical significance

---

### Step 8: Cross-Validation

**File:** [src/evaluation/cross_validation.py](src/evaluation/cross_validation.py)

```bash
.venv/bin/python src/evaluation/cross_validation.py
```

**Purpose:** Verify that LR and SVM results are stable — not dependent on the particular 70/15/15 split.

**Method:** 5-fold stratified CV × 3 random seeds = **15 folds per model**

```
Training data (9,770 examples)
├── Fold 1: [Val: 20%=1954] + [Train: 80%=7816]  → compute F1
├── Fold 2: [Val: 20%]      + [Train: 80%]        → compute F1
├── Fold 3: [Val: 20%]      + [Train: 80%]        → compute F1
├── Fold 4: [Val: 20%]      + [Train: 80%]        → compute F1
└── Fold 5: [Val: 20%]      + [Train: 80%]        → compute F1
Repeat with 3 different random seeds → average 15 F1 scores ± std
```

A small std (≤ ±0.01) means the model is **robust** — performance doesn't depend heavily on which examples end up in train vs validation.

#### What to Learn

- K-fold cross-validation
- Why CV gives more reliable estimates than a single split
- Mean ± std as a way to report model stability

---

### Step 9: Ablation Study

**File:** [src/evaluation/ablation_study.py](src/evaluation/ablation_study.py)

```bash
.venv/bin/python src/evaluation/ablation_study.py
```

**Purpose:** Justify each design decision by testing what happens when you remove or change it. The rule: **change one thing at a time**.

**5 ablations:**

| What changes | Values tested | What we learn |
|---|---|---|
| Vocabulary size (`max_features`) | 1K, 5K, 10K, 20K, 40K | 40K is the sweet spot |
| N-gram range | unigrams only vs +(1,2) bigrams | Bigrams improve F1 (~+1%) |
| Word segmentation | with RDR vs without | Segmentation adds +2–3% F1 |
| Regularization strength (C) | 0.001, 0.01, 0.1, 1, 10, 100 | C=10 is optimal |
| TF scaling | standard count vs sublinear | `log(1+tf)` is slightly better |

This lets you tell reviewers: *"every design choice was validated quantitatively."*

#### What to Learn

- Ablation study methodology: one variable at a time
- How to isolate the contribution of each component

---

### Step 10: Explainability

**File:** [src/analysis/explainability.py](src/analysis/explainability.py)

```bash
.venv/bin/python src/analysis/explainability.py
```

**Purpose:** Make the model interpretable — which words push toward Real vs Fake?

**Method:** Logistic Regression stores learned weights `coef_[class, feature_idx]`. The weight for (class=Fake, feature="giả_mạo") directly tells you how much the word "giả_mạo" shifts probability toward Fake.

**Why LR and not PhoBERT for explainability?**

| Model | Interpretability |
|---|---|
| Logistic Regression | `w_i` directly = contribution of word `i` → fully interpretable |
| BiLSTM | attention weights ≈ importance, but not a complete explanation |
| PhoBERT | black box — no simple mapping from input to output |

For black-box neural models, the standard approach is **LIME** (Local Interpretable Model-agnostic Explanations) or **SHAP** — not in this project but worth knowing.

**Output:**
- `results/figures/explainability/feature_importance.{png,pdf}`
- `paper/figures/feature_importance.{png,pdf}`

#### What to Learn

- Linear model coefficient interpretation
- Model interpretability vs explainability
- Black box vs interpretable models
- LIME / SHAP (for presentation awareness)

---

### Step 11–12: Paper Figures & Tables

**Files:** [src/analysis/generate_paper_figures.py](src/analysis/generate_paper_figures.py), [src/analysis/generate_paper_tables.py](src/analysis/generate_paper_tables.py)

```bash
.venv/bin/python src/analysis/generate_paper_figures.py
.venv/bin/python src/analysis/generate_paper_tables.py
```

Formats all results for publication:
- **Figures:** 300 DPI, PDF format → `paper/figures/fig{1-6}_*.{png,pdf}`
- **Tables:** LaTeX `.tex` files for `\input{}` → `paper/tables/table{1-5}_*.tex`

| Figure | Content |
|---|---|
| fig1 | Model comparison bar chart |
| fig2 | Confusion matrices (2×4 grid) |
| fig3 | ROC curves (all 4 models) |
| fig4 | Precision-Recall curves |
| fig5 | Per-class (Real vs Fake) breakdown |
| fig6 | Paradigm comparison (classical vs deep vs transformer) |

| Table | Content |
|---|---|
| table1 | Dataset statistics |
| table2 | Main results (primary contribution) |
| table3 | Per-class metrics |
| table4 | Hyperparameters |
| table5 | Computational complexity |

---

## 4. Concepts You Need to Understand

### Level 1 — Must Know for Presentation

| Concept | Key Point | File to Read |
|---|---|---|
| **Vietnamese word segmentation** | Compound words need joining; same tokenizer for all models | [src/preprocessing/word_segmentation.py](src/preprocessing/word_segmentation.py) |
| **TF-IDF** | Rare-but-present words are most informative; 40K-dim sparse vector | [src/features/tfidf_features.py](src/features/tfidf_features.py) |
| **Train/Val/Test split** | 3 separate sets; test touched only once at the end | [src/preprocessing/split_data.py](src/preprocessing/split_data.py) |
| **Stratified sampling** | Preserve class ratio in each split | `split_data()` function |
| **Precision / Recall / F1** | F1 = harmonic mean; macro = equal weight per class | [src/evaluation/metrics.py](src/evaluation/metrics.py) |
| **Confusion matrix** | TP/TN/FP/FN — visualize error types | `plot_confusion_matrix()` in metrics.py |
| **Logistic Regression** | Linear model; C controls regularization | [src/training/train_lr.py](src/training/train_lr.py) |
| **LinearSVC** | Margin maximization; uses CalibratedClassifierCV for probabilities | [src/training/train_svm.py](src/training/train_svm.py) |
| **GridSearchCV** | Exhaustive hyperparameter tuning using cross-validation | `tune_hyperparameters()` in train_lr/svm.py |

### Level 2 — Intermediate (Good to Know)

| Concept | Key Point | File to Read |
|---|---|---|
| **Word embeddings** | Dense 300-dim vectors; similar words have similar vectors | [src/features/embedding_features.py](src/features/embedding_features.py) |
| **FastText pretrained** | Pretrained on Vietnamese Wikipedia+CommonCrawl; subword-aware | `cfg.BILSTM.fasttext_path` |
| **LSTM / BiLSTM** | Maintains hidden state; reads both forward and backward | [src/models/bilstm_model.py](src/models/bilstm_model.py) |
| **Soft Attention** | Weighted average over timesteps; `α_t` = importance of word t | `self.attention` in bilstm_model.py |
| **Early stopping** | Stop when val F1 stops improving; prevents overfitting | `EarlyStopping` class in train_bilstm.py |
| **Class imbalance** | `class_weight="balanced"` upweights minority class | `compute_class_weight()` usage |
| **K-fold CV** | More reliable evaluation than single train/test split | [src/evaluation/cross_validation.py](src/evaluation/cross_validation.py) |

### Level 3 — Research-Level

| Concept | Key Point | File to Read |
|---|---|---|
| **BERT / Transformer** | 12-layer self-attention; every token attends to every other | [src/models/phobert_model.py](src/models/phobert_model.py) |
| **Fine-tuning pretrained** | Low LR=3e-5; layer LR decay protects early layers | [src/training/train_phobert.py](src/training/train_phobert.py) |
| **Gradient accumulation** | Effective batch 64 without large GPU; steps every 4 batches | Training loop in train_phobert.py |
| **LR warmup + linear decay** | Stable BERT training; prevents early divergence | `get_linear_schedule_with_warmup` |
| **McNemar's test** | Sample-wise comparison; tests if error distributions differ | [src/analysis/statistical_tests.py](src/analysis/statistical_tests.py) |
| **Bootstrap CI** | Non-parametric 95% CI via 10,000 resamples | `bootstrap_confidence_interval()` |
| **Holm-Bonferroni** | Controls family-wise error rate for 6 pairwise tests | `holm_bonferroni_correction()` |
| **Ablation study** | Change one component at a time to measure its contribution | [src/evaluation/ablation_study.py](src/evaluation/ablation_study.py) |
| **Feature importance** | LR coefficient `w_i` = contribution of word i to log-odds | [src/analysis/explainability.py](src/analysis/explainability.py) |

### Level 4 — Engineering

| Concept | Key Point | File to Read |
|---|---|---|
| **Central config pattern** | All hyperparams in one `@dataclass`; import `cfg` everywhere | [config.py](config.py) |
| **`sys.path` pattern** | `src/` scripts add project root to path for cross-module imports | Top 15 lines of any `src/` file |
| **`pickle` vs `torch.save`** | sklearn models saved with pickle; PyTorch with `torch.save` | `src/training/train_*.py` |
| **PyTorch `Dataset`** | Custom class holding data; iterated by DataLoader | `TextDataset(Dataset)` in embedding_features.py |
| **DataLoader** | Batches + shuffles Dataset; controls training data flow | `create_data_loaders()` |
| **Logging** | Structured output to console + file; `get_logger(__name__)` | [src/utils/logger.py](src/utils/logger.py) |

---

## 5. Model Architecture Deep-Dive

### Why 4 Models?

| Model | Paradigm | Input Features | Key Strength | Key Weakness |
|---|---|---|---|---|
| Logistic Regression | Classical ML | TF-IDF (40K-dim sparse) | Fast; coefficients are directly interpretable | Linear only; ignores word order |
| LinearSVC | Classical ML | TF-IDF (40K-dim sparse) | Margin maximization; robust for text | No probability output natively |
| BiLSTM | Deep Learning | FastText embeddings (300-dim) | Captures word order; pretrained embeddings | Sequential; needs more data than 10K |
| PhoBERT | Transformer | BPE subwords (64K vocab); 256 tokens | 20GB Vietnamese pretraining; global attention | Slow; GPU required; ~125M parameters |

### Key Conceptual Differences

**Classical ML (LR, SVM) — Bag of Words:**
```
"The president fired the minister" == "The minister fired the president"
Both produce IDENTICAL 40K-dim TF-IDF vectors because word order is ignored.
```

**BiLSTM — Sequential Context:**
```
"giả_mạo bài viết" → LSTM reads left→right, remembers "giả_mạo" when it sees "bài_viết"
Word order matters; BUT long-range dependencies can be forgotten.
```

**PhoBERT — Global Self-Attention:**
```
Every word attends to every other word simultaneously.
"Tổng_thống [MASK] ký hiệp_ước" — both "Tổng_thống" and "hiệp_ước" help predict [MASK].
No sequential processing; no forgetting.
```

### Why `lr=3e-5` for PhoBERT but `lr=1e-3` for BiLSTM?

BiLSTM starts from scratch (random weights). We need a larger step size to quickly find good solutions.

PhoBERT starts from excellent pretrained weights (learned from 20GB Vietnamese text). We make tiny adjustments:
- Too large LR (e.g. 1e-3) → **catastrophic forgetting**: the gradients overwrite the pretrained knowledge in a few steps
- LR=3e-5 → tiny steps → we keep the pretrained knowledge and nudge it toward fake news

This is **transfer learning** — the most important concept in modern NLP.

---

## 6. Actual Results

> All numbers below come directly from `experiments/*/metrics.json` — verified against the actual files.

### Final Test Set Results

| Model | Accuracy | F1 (macro) | ROC-AUC | Best Params |
|---|---|---|---|---|
| **PhoBERT** | **90.1%** | **89.9%** | **95.0%** | `vinai/phobert-base`, lr=3e-5, batch=16 |
| LinearSVC | 84.3% | 84.1% | 91.9% | LinearSVC, C=0.5 |
| LogReg | 83.5% | 83.3% | 91.9% | C=10, saga, l2 |
| BiLSTM | 82.5% | 82.3% | 90.5% | hidden=128, FastText 300-dim |

### Per-Class Results on Test Set — PhoBERT

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Thật / Real (0) | 89.4% | 93.1% | 91.3% | 1,165 |
| Giả / Fake (1) | 90.9% | 86.2% | 88.5% | 929 |

**Interpretation:** PhoBERT correctly identifies Real news more reliably (higher recall=93.1%) than it catches all Fake news (recall=86.2%). About 13.8% of fake articles slip through as "real". This is a known limitation of the model.

### Per-Class Results on Test Set — BiLSTM

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Thật / Real (0) | 84.6% | 83.9% | 84.2% | 1,165 |
| Giả / Fake (1) | 80.0% | 80.8% | 80.4% | 929 |

### Per-Class Results on Test Set — LogReg

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Thật / Real (0) | 85.5% | 84.6% | 85.1% | 1,165 |
| Giả / Fake (1) | 81.0% | 82.0% | 81.5% | 929 |

### Per-Class Results on Test Set — LinearSVC

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Thật / Real (0) | 85.4% | 86.7% | 86.0% | 1,165 |
| Giả / Fake (1) | 83.0% | 81.4% | 82.2% | 929 |

### Training History

| Model | Epochs trained | Best val F1 | Notes |
|---|---|---|---|
| PhoBERT | 8 | 90.8% | patience=2 |
| BiLSTM | 8 | 83.6% | patience=5 |
| LR | N/A | 84.5% (val) | GridSearchCV across 5 folds |
| SVM | N/A | 83.3% (val) | GridSearchCV across 5 folds |

### Dataset Statistics (verified from actual CSV files)

| Metric | Value |
|---|---|
| Raw articles (`raw.csv`) | 16,946 |
| After segmentation (`segmented.csv`) | 15,789 |
| After cleaning (≥10 words) | ~13,958 |
| **Train split** | **9,770** |
| **Val split** | **2,094** |
| **Test split** | **2,094** |
| Test: Real (Thật, label=0) | 1,165 (55.6%) |
| Test: Fake (Giả, label=1) | 929 (44.4%) |

---

## 7. Quick Reference

### "I want to understand..." → "Read this file"

| Topic | File |
|---|---|
| All hyperparameters | [config.py](config.py) |
| How Vietnamese text is processed | [src/preprocessing/word_segmentation.py](src/preprocessing/word_segmentation.py) |
| How text → TF-IDF numbers | [src/features/tfidf_features.py](src/features/tfidf_features.py) |
| How text → word IDs for BiLSTM | [src/features/embedding_features.py](src/features/embedding_features.py) |
| How text → PhoBERT tokens | [src/features/phobert_features.py](src/features/phobert_features.py) |
| BiLSTM architecture | [src/models/bilstm_model.py](src/models/bilstm_model.py) |
| PhoBERT architecture | [src/models/phobert_model.py](src/models/phobert_model.py) |
| LR training + hyperparameter tuning | [src/training/train_lr.py](src/training/train_lr.py) |
| SVM training + hyperparameter tuning | [src/training/train_svm.py](src/training/train_svm.py) |
| BiLSTM training loop | [src/training/train_bilstm.py](src/training/train_bilstm.py) |
| PhoBERT fine-tuning loop | [src/training/train_phobert.py](src/training/train_phobert.py) |
| How F1-score is calculated | [src/evaluation/metrics.py](src/evaluation/metrics.py) |
| Why PhoBERT is better (statistically) | [src/analysis/statistical_tests.py](src/analysis/statistical_tests.py) |
| Which words indicate fake news | [src/analysis/explainability.py](src/analysis/explainability.py) |
| How to run the full pipeline | [MANUAL.md](MANUAL.md) |

### Presentation Q&A Cheat Sheet

**Q: Why four different models?**  
A: To show a clear progression — classical ML (LR, SVM) → sequential deep learning (BiLSTM) → transformer (PhoBERT). Each level shows how much more Vietnamese language understanding you gain from more data and compute.

**Q: Why not use PhoBERT for everything?**  
A: Training cost. PhoBERT takes ~30–60 min on GPU per run. LR/SVM train in seconds. For production with strict latency requirements, 84% accuracy in 0.01s beats 90% in 1s.

**Q: Why does BiLSTM (82.3%) underperform LinearSVC (84.1%) despite being more complex?**  
A: Complexity needs data. With ~10K training examples, LSTM's gains from modeling word order are outweighed by its tendency to overfit. Classical models with good sparse features often beat deep models on small datasets.

**Q: How do you know PhoBERT is genuinely better, not just lucky?**  
A: McNemar's test with Holm-Bonferroni correction gives p < 0.001. The probability that the 89.9% vs 83.3% difference is due to random chance is less than 0.1%.

**Q: Why do LR and BiLSTM score similarly?**  
A: McNemar's test shows p ≈ 0.94 between them — they are statistically **not different**. Different architectures, same result. This is a meaningful finding that LR with good features can match a more complex sequential model on this dataset.

**Q: What is fine-tuning?**  
A: Start from a model (PhoBERT) already trained on 20GB of Vietnamese text (it understands Vietnamese). Then train it for a few epochs on YOUR specific task (fake news) with a very small learning rate (3e-5) so you don't destroy the pretrained knowledge. Result: deep language understanding + task specialization. This is called **transfer learning**.

**Q: What is word segmentation and why is it first?**  
A: Vietnamese compound words use spaces between syllables, but they're one semantic unit. Without segmentation, "thành phố" (city) is split into "thành" (complete) + "phố" (street), destroying meaning. We must segment BEFORE building any features.

**Q: Why use the same tokenizer (VnCoreNLP RDRSegmenter) for all models?**  
A: PhoBERT's vocabulary was built from RDRSegmenter output. If you use a different segmenter for PhoBERT features, the token shapes won't match the model's vocabulary and performance degrades. For fairness in comparison, all models use the same tokenizer.

**Q: What is TF-IDF and why does it work for fake vs real news?**  
A: TF-IDF assigns high scores to words that appear often in a specific article but are rare across all articles. Fake news tends to use distinctive vocabulary (sensationalist words, fake citation phrases) while real news uses specific names and facts. TF-IDF captures these discriminative features well.

**Q: What is gradient accumulation?**  
A: Instead of computing one gradient update per batch, you accumulate gradients across N batches then step once. With `batch_size=16` and `accumulation_steps=4`, every 4 forward passes count as one "effective batch" of 64. This lets you simulate large batches without large GPU memory.

**Q: What is layer LR decay in PhoBERT?**  
A: Different transformer layers learn different things. Early layers (close to input) learn basic language structure — we want to preserve this. Later layers learn task-specific patterns — we want to adapt these more. `layer_lr_decay=0.95` means each layer below the top gets LR × 0.95, protecting deeper pretrained knowledge.

---

*Last updated: verified against `experiments/*/metrics.json`, `config.py`, `src/models/*.py`, and actual data CSV files.*
