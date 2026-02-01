# Vietnamese Fake News Detection - Research Pipeline Summary

This document summarizes all the work completed across the 8-step research pipeline.

---

## Step 1: Data Collection ✅

**Objective:** Gather Vietnamese news articles labeled as real or fake.

**What was done:**

- Collected raw Vietnamese news data stored in `data/raw/raw.csv`
- Dataset contains news articles with fields: `id`, `title`, `text`, `label`
- Labels: 0 = Real News, 1 = Fake News

**Output:**

- `data/raw/raw.csv` - Original raw dataset

---

## Step 2: Data Exploration ✅

**Objective:** Understand the dataset characteristics and identify issues.

**What was done:**

- Analyzed dataset statistics and distribution
- Discovered critical issues:
  - **Data leakage**: 392 overlapping texts between train/val/test splits
  - **Duplicates**: 433 duplicate texts across datasets
  - **Class imbalance**: ~63% Real, ~37% Fake

**Key findings:**

- Total samples before cleaning: 6,497
- Duplicate records identified and flagged for removal
- Class distribution slightly imbalanced but manageable

---

## Step 3: Text Preprocessing ✅

**Objective:** Clean and prepare text data for model training.

**What was done:**

- Vietnamese word segmentation using VnCoreNLP
- Text stored with underscore-connected compound words (e.g., "Việt_Nam", "bài_thuốc")
- Text cleaning: URL removal, special character handling, whitespace normalization

**Files created:**

- `src/preprocessing/text_preprocessor.py` - Text cleaning utilities
- `src/preprocessing/word_segmentation.py` - Vietnamese word segmentation
- `data/processed/segmented.csv` - Preprocessed text data

---

## Step 4: Data Splitting ✅

**Objective:** Create clean train/validation/test splits without data leakage.

**What was done:**

- Removed all duplicate texts
- Removed empty and very short texts (< 5 words)
- Created stratified splits maintaining class distribution
- Split ratio: 70% train / 15% validation / 15% test

**Results after cleaning:**
| Split | Total | Real (0) | Fake (1) |
|-------|-------|----------|----------|
| Train | 3,957 | 2,490 | 1,467 |
| Val | 849 | 534 | 315 |
| Test | 849 | 534 | 315 |

**Files created/updated:**

- `src/preprocessing/split_data.py` - Data cleaning and splitting logic
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

---

## Step 5: Feature Engineering ✅

**Objective:** Extract features from text for different model types.

**What was done:**

### TF-IDF Features (for LR & SVM)

- Vocabulary size: 10,000 terms
- N-gram range: (1, 2) - unigrams and bigrams
- Sublinear TF scaling enabled
- Feature dimensions: (3957, 10000)

### Word Embeddings (for BiLSTM)

- Built vocabulary from training data
- Vocabulary size: 22,852 unique words
- Embedding dimension: 128
- Maximum sequence length: 200 tokens

### PhoBERT Tokenization (for Transformer)

- Using `vinai/phobert-base` tokenizer
- Maximum sequence length: 256 tokens
- Attention masks generated

**Files created:**

- `src/features/__init__.py`
- `src/features/tfidf_features.py` - TF-IDF feature extractor
- `src/features/embedding_features.py` - Word embedding & vocabulary builder
- `src/features/phobert_features.py` - PhoBERT tokenization
- `src/features/extract_all_features.py` - Feature extraction orchestrator

**Output:**

- `data/features/tfidf/tfidf_features.pkl`
- `data/features/embedding/embedding_features.pkl`
- `data/features/phobert/phobert_features.pkl`

---

## Step 6: Model Training ✅

**Objective:** Train 4 models representing different ML paradigms.

**Models trained:**

### 1. Logistic Regression (Traditional ML)

- **Features:** TF-IDF
- **Hyperparameters:** C=10, solver=lbfgs, max_iter=1000
- **Training time:** ~3 seconds
- **Validation F1:** 0.8644

### 2. Support Vector Machine (Traditional ML)

- **Features:** TF-IDF
- **Hyperparameters:** C=10, kernel=rbf, gamma=scale
- **Training time:** ~119 seconds
- **Validation F1:** 0.8644

### 3. BiLSTM (Deep Learning)

- **Features:** Word embeddings
- **Architecture:** 2-layer Bidirectional LSTM, hidden_dim=128, dropout=0.3
- **Training:** Adam optimizer, lr=0.001, early stopping (patience=3)
- **Training time:** ~18 seconds
- **Best Validation F1:** 0.8652

### 4. PhoBERT (Transformer)

- **Model:** `vinai/phobert-base` (pre-trained)
- **Architecture:** BERT + linear classifier
- **Training:** AdamW optimizer, lr=2e-5, 5 epochs
- **Training time:** ~311 seconds
- **Best Validation F1:** 0.8961

**Files created:**

- `src/training/train_lr.py` - Logistic Regression training
- `src/training/train_svm.py` - SVM training
- `src/training/train_bilstm.py` - BiLSTM training
- `src/training/train_phobert.py` - PhoBERT training
- `src/training/train_all.py` - Training orchestrator

**Output:**

- `experiments/lr/lr_model.pkl` + `metrics.json`
- `experiments/svm/svm_model.pkl` + `metrics.json`
- `experiments/bilstm/bilstm_model.pt` + `metrics.json`
- `experiments/bert/phobert_model.pt` + `metrics.json`

---

## Step 7: Evaluation & Analysis ✅

**Objective:** Comprehensive evaluation and comparison of all models.

**What was done:**

### Performance Comparison

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **PhoBERT** 🏆      | **90.58%** | **91.14%** | **88.60%** | **89.61%** | **95.78%** |
| SVM                 | 87.63%     | 87.53%     | 85.68%     | 86.44%     | 94.84%     |
| BiLSTM              | 87.28%     | 86.18%     | 86.96%     | 86.52%     | 93.85%     |
| Logistic Regression | 87.28%     | 86.27%     | 86.63%     | 86.44%     | 94.19%     |

### Error Analysis

- PhoBERT: Lowest error rate (9.42%)
- 29 samples misclassified by ALL models (hard examples)
- Shorter texts tend to be harder to classify
- PhoBERT excels at reducing false positives

**Files created:**

- `src/evaluation/evaluate_all.py` - Comprehensive evaluation
- `src/evaluation/error_analysis.py` - Error pattern analysis

**Output:**

- `results/figures/model_comparison.png`
- `results/figures/confusion_matrices.png`
- `results/figures/roc_curves.png`
- `results/figures/training_history.png`
- `results/figures/error_analysis/` - Error analysis visualizations
- `results/tables/model_comparison.csv`
- `results/tables/per_class_metrics.csv`
- `results/tables/hard_examples.csv`
- `results/tables/evaluation_summary.json`

---

## Step 8: Result Analysis & Paper Writing ✅

**Objective:** Statistical analysis and paper preparation.

**What was done:**

### Statistical Significance Testing

- **McNemar's Test:** PhoBERT vs all others is significant (p < 0.05)
- **Bootstrap Confidence Intervals:** 95% CI for all metrics
- **Effect Size (Cohen's d):** Negligible individual effect, but consistent improvement

### Key Statistical Results

| Comparison        | χ²   | p-value | Significant |
| ----------------- | ---- | ------- | ----------- |
| PhoBERT vs LR     | 6.51 | 0.011   | ✅ Yes      |
| PhoBERT vs SVM    | 6.33 | 0.012   | ✅ Yes      |
| PhoBERT vs BiLSTM | 6.39 | 0.011   | ✅ Yes      |

### Paper Materials Generated

**Figures (PNG + PDF):**

- `fig1_model_comparison` - Performance bar chart
- `fig2_confusion_matrices` - 4-panel confusion matrices
- `fig3_roc_curves` - ROC curves comparison
- `fig4_pr_curves` - Precision-Recall curves
- `fig5_per_class` - Per-class performance
- `fig6_paradigm_comparison` - Paradigm comparison

**LaTeX Tables:**

- `table1_dataset.tex` - Dataset statistics
- `table2_results.tex` - Main results
- `table3_perclass.tex` - Per-class metrics
- `table4_hyperparams.tex` - Hyperparameters
- `table5_complexity.tex` - Model complexity

**Files created:**

- `src/analysis/__init__.py`
- `src/analysis/statistical_tests.py` - Statistical analysis
- `src/analysis/generate_paper_figures.py` - Figure generation
- `src/analysis/generate_paper_tables.py` - Table generation
- `paper/paper_draft.md` - Full paper draft
- `paper/main.tex` - LaTeX template
- `paper/figures/` - Publication-ready figures
- `paper/tables/` - LaTeX tables

---

## Final Project Structure

```
FakeNewsDetector/
├── data/
│   ├── raw/raw.csv                    # Original data
│   ├── processed/segmented.csv        # Preprocessed data
│   ├── splits/                        # Train/Val/Test splits
│   └── features/                      # Extracted features
├── src/
│   ├── preprocessing/                 # Text preprocessing
│   ├── features/                      # Feature extraction
│   ├── training/                      # Model training
│   ├── evaluation/                    # Model evaluation
│   └── analysis/                      # Statistical analysis
├── experiments/
│   ├── lr/                            # LR model & metrics
│   ├── svm/                           # SVM model & metrics
│   ├── bilstm/                        # BiLSTM model & metrics
│   └── bert/                          # PhoBERT model & metrics
├── results/
│   ├── figures/                       # Evaluation visualizations
│   └── tables/                        # Result tables
├── paper/
│   ├── figures/                       # Publication figures
│   ├── tables/                        # LaTeX tables
│   ├── paper_draft.md                 # Paper draft
│   └── main.tex                       # LaTeX template
└── notes/
    └── research_pipeline_summary.md   # This file
```

---

## Key Takeaways

1. **Best Model:** PhoBERT achieves 90.58% accuracy, significantly outperforming all baselines
2. **Improvement:** +3.7% over traditional ML baseline (statistically significant, p < 0.05)
3. **Trade-off:** PhoBERT requires more computational resources but delivers superior accuracy
4. **Practical Insight:** Traditional ML (LR/SVM) remains viable for resource-constrained scenarios

---

## Next Steps (Future Work)

- [ ] Multimodal detection (incorporate images)
- [ ] Explainability analysis (attention visualization)
- [ ] Cross-domain evaluation (different news sources)
- [ ] Larger dataset collection
- [ ] Real-time deployment API

---

_Document generated: February 1, 2026_
