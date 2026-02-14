# Vietnamese Fake News Detection - Research Pipeline Summary

This document summarizes all the work completed across the research pipeline.

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

---

## Step 3: Text Preprocessing ✅

**Objective:** Clean and prepare text data for model training.

**What was done:**

- Vietnamese word segmentation using VnCoreNLP
- Text cleaning: URL removal, special character handling, whitespace normalization

**Files:**

- `src/preprocessing/text_preprocessor.py`
- `src/preprocessing/word_segmentation.py`
- `data/processed/segmented.csv`

---

## Step 4: Data Splitting ✅

**Objective:** Create clean train/validation/test splits without data leakage.

**Results after cleaning:**
| Split | Total | Real (0) | Fake (1) |
|-------|-------|----------|----------|
| Train | 3,957 | 2,490 | 1,467 |
| Val | 849 | 534 | 315 |
| Test | 849 | 534 | 315 |

---

## Step 5: Feature Engineering ✅

**Features extracted:**

- **TF-IDF**: 10,000 terms, (1,2) n-grams, sublinear TF
- **Word Embeddings**: vocab=22,852, dim=256, max_len=200
- **PhoBERT Tokenization**: phobert-base, max_len=256

---

## Step 6: Model Training ✅

| Model                                 | Features       | Best Val F1 | Training Time |
| ------------------------------------- | -------------- | ----------- | ------------- |
| LR (C=10, lbfgs)                      | TF-IDF         | 0.8792      | ~3s           |
| SVM (C=10, rbf)                       | TF-IDF         | 0.8841      | ~119s         |
| BiLSTM (2-layer, emb=256, hidden=128) | Embeddings     | 0.8676      | ~18s          |
| PhoBERT (phobert-base, lr=2e-5)       | Subword tokens | 0.9022      | ~311s         |

---

## Step 7: Evaluation & Analysis ✅

### Test Set Performance

| Model       | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **PhoBERT** | **90.58%** | **91.14%** | **88.60%** | **89.61%** | **95.78%** |
| SVM         | 87.63%     | 87.53%     | 85.68%     | 86.44%     | 94.84%     |
| BiLSTM      | 87.28%     | 86.18%     | 86.96%     | 86.52%     | 93.85%     |
| LR          | 87.28%     | 86.27%     | 86.63%     | 86.44%     | 94.19%     |

---

## Step 8: Statistical Analysis ✅ (Updated)

### McNemar's Test with Holm-Bonferroni Correction

Applied to control family-wise error rate across 6 pairwise comparisons.

| Comparison        | χ²   | p (raw) | p (adjusted) | Significant |
| ----------------- | ---- | ------- | ------------ | ----------- |
| PhoBERT vs LR     | 6.51 | 0.011   | 0.064        | No†         |
| PhoBERT vs SVM    | 6.33 | 0.012   | 0.064        | No†         |
| PhoBERT vs BiLSTM | 6.39 | 0.011   | 0.064        | No†         |
| LR vs SVM         | 0.09 | 0.771   | 1.000        | No          |
| LR vs BiLSTM      | 0.01 | 0.918   | 1.000        | No          |
| SVM vs BiLSTM     | 0.04 | 0.842   | 1.000        | No          |

†Marginally non-significant at α = 0.05 after Holm-Bonferroni correction (raw p ≈ 0.011)

### Bootstrap Confidence Intervals (95%, n=10,000)

Increased from 1,000 to 10,000 iterations (modern standard).

### Effect Size (Cohen's d)

Reported for all PhoBERT vs baseline comparisons. Effect sizes are negligible at individual sample level but aggregate improvement is practically meaningful.

---

## Step 9: Cross-Validation ✅ (NEW)

**Objective:** Provide robust performance estimates beyond single train/test split.

**What was done:**

- Stratified 5-fold CV with 3 random seeds (15 total folds) for LR and SVM
- Combines train + validation sets for full CV evaluation

**Files:**

- `src/evaluation/cross_validation.py`
- Output → `results/tables/cross_validation_results.json`
- Output → `results/tables/cross_validation_summary.csv`

---

## Step 10: Ablation Study ✅ (NEW)

**Objective:** Quantify the contribution of key design choices.

**Ablations performed:**

1. **TF-IDF vocabulary size**: 1K, 5K, 10K, 20K, 50K terms
2. **N-gram range**: unigrams, bigrams, trigrams, combinations
3. **Word segmentation**: with vs without VnCoreNLP segmentation
4. **Regularization strength**: C ∈ {0.01, 0.1, 1, 10, 100}
5. **Sublinear TF scaling**: standard vs sublinear

**Files:**

- `src/evaluation/ablation_study.py`
- Output → `results/tables/ablation_study.json`
- Output → `results/tables/ablation_study.tex`

---

## Step 11: Explainability Analysis ✅ (NEW)

**Objective:** Provide interpretability insights for model predictions.

**What was done:**

1. **LR Feature Importance**: Top 30 predictive words for Real vs Fake news using model coefficients
2. **Error Categorization Taxonomy**: Errors classified by text length (short/medium/long) and confidence level (high/low)
3. **Visualizations**: Feature importance plots, error taxonomy charts

**Files:**

- `src/analysis/explainability.py`
- Output → `results/figures/explainability/feature_importance.png`
- Output → `results/figures/explainability/error_taxonomy.png`
- Output → `paper/figures/feature_importance.pdf`
- Output → `paper/figures/error_taxonomy.pdf`

---

## Step 12: Paper Writing ✅ (Updated)

### Paper improvements:

1. **Expanded Related Work**: ~2 pages with 25 references (previously 4)
2. **Complete LaTeX paper** (`paper/main.tex`): All sections, figures, tables, references
3. **Holm-Bonferroni correction**: Properly reported in paper
4. **Ablation study section**: New section in paper
5. **Interpretability section**: New section in paper
6. **Cross-validation results**: Reported for robust estimation
7. **Effect size analysis**: Cohen's d reported with proper interpretation
8. **Fixed inconsistencies**:
   - BiLSTM embedding_dim correctly reported as 256
   - Training times filled in Table 5
   - Python version correctly as 3.14
   - Repository URL: https://github.com/hoanganh0705/FakeNewsDetector

---

## Reproducibility Improvements ✅ (NEW)

1. **requirements.txt**: All packages pinned with exact versions
2. **README.md**: Full installation, usage, and results documentation
3. **Code availability**: GitHub URL specified
4. **Random seed**: 42 documented throughout

---

## Final Project Structure

```
FakeNewsDetector/
├── data/
│   ├── raw/raw.csv
│   ├── processed/segmented.csv
│   ├── splits/{train,val,test}.csv
│   └── features/{tfidf,embedding,phobert}/
├── src/
│   ├── preprocessing/          # Text cleaning, segmentation, splitting
│   ├── features/               # TF-IDF, embeddings, PhoBERT tokenization
│   ├── training/               # LR, SVM, BiLSTM, PhoBERT training
│   ├── evaluation/
│   │   ├── evaluate_all.py     # Comprehensive evaluation
│   │   ├── error_analysis.py   # Error pattern analysis
│   │   ├── cross_validation.py # 5-fold CV (NEW)
│   │   ├── ablation_study.py   # Ablation studies (NEW)
│   │   └── metrics.py          # Metric utilities
│   ├── analysis/
│   │   ├── statistical_tests.py    # McNemar + Holm-Bonferroni (UPDATED)
│   │   ├── explainability.py       # Feature importance (NEW)
│   │   ├── generate_paper_figures.py
│   │   └── generate_paper_tables.py
│   └── utils/
├── experiments/{lr,svm,bilstm,bert}/
├── results/{figures,tables}/
├── paper/
│   ├── main.tex               # Complete LaTeX paper (UPDATED)
│   ├── paper_draft.md         # Full paper draft (UPDATED)
│   ├── figures/               # 8 publication figures (6 + 2 NEW)
│   └── tables/                # 5 LaTeX tables (UPDATED)
├── requirements.txt           # Pinned versions (UPDATED)
├── README.md                  # Full documentation (NEW)
└── notes/
    └── research_pipeline_summary.md
```

---

## Key Takeaways

1. **Best Model:** PhoBERT achieves 90.58% accuracy, numerically outperforming all baselines
2. **Statistical Rigor:** Holm-Bonferroni correction reveals marginal non-significance (p_adj=0.064), demonstrating importance of multiple comparison correction
3. **Robust Estimation:** 5-fold CV with 3 seeds confirms single-split results for traditional ML
4. **Word Segmentation:** Ablation shows minimal impact with TF-IDF features on this dataset
5. **Practical Trade-off:** PhoBERT (135M params, 311s) vs LR (10K params, 3s) for 3% accuracy gain

---

_Document updated: February 13, 2026_
