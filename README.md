# Vietnamese Fake News Detection: A Comparative Study of Machine Learning Approaches

A comprehensive research project comparing machine learning approaches for Vietnamese fake news detection, evaluating four models across different paradigms: traditional ML (Logistic Regression, SVM), deep learning (BiLSTM), and transformer-based (PhoBERT).

## Key Results

| Model               | Accuracy   | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- |
| **PhoBERT**         | **90.58%** | **0.8961** | **0.9578** |
| SVM                 | 87.63%     | 0.8644     | 0.9484     |
| BiLSTM              | 87.28%     | 0.8652     | 0.9385     |
| Logistic Regression | 87.28%     | 0.8644     | 0.9419     |

PhoBERT significantly outperforms all baselines (McNemar's test with Holm-Bonferroni correction).

## Project Structure

```
FakeNewsDetector/
├── data/
│   ├── raw/raw.csv                     # Original dataset
│   ├── processed/segmented.csv         # Word-segmented text
│   ├── splits/                         # Train/Val/Test splits (70/15/15)
│   └── features/                       # Extracted features (TF-IDF, embeddings, PhoBERT)
├── src/
│   ├── preprocessing/                  # Text cleaning & word segmentation
│   ├── features/                       # Feature extraction (TF-IDF, embeddings, PhoBERT)
│   ├── training/                       # Model training scripts
│   ├── evaluation/                     # Metrics, error analysis, cross-validation, ablation
│   └── analysis/                       # Statistical tests, explainability, paper generation
├── experiments/                        # Trained models & metrics
├── results/                            # Figures, tables, evaluation outputs
├── paper/                              # LaTeX paper, figures, tables
└── notes/                              # Research notes & pipeline summary
```

## Installation

### Prerequisites

- Python 3.14+
- CUDA-compatible GPU (recommended for PhoBERT/BiLSTM training)

### Setup

```bash
git clone https://github.com/hoanganh0705/FakeNewsDetector.git
cd FakeNewsDetector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
# Word segmentation
python src/preprocessing/word_segmentation.py

# Data cleaning and splitting
python src/preprocessing/split_data.py
```

### 2. Feature Extraction

```bash
python src/features/extract_all_features.py
```

### 3. Model Training

```bash
# Train all models
python src/training/train_all.py

# Or train individually
python src/training/train_lr.py
python src/training/train_svm.py
python src/training/train_bilstm.py
python src/training/train_phobert.py
```

### 4. Evaluation

```bash
# Comprehensive evaluation with visualizations
python src/evaluation/evaluate_all.py

# Cross-validation for traditional ML models
python src/evaluation/cross_validation.py

# Ablation study
python src/evaluation/ablation_study.py

# Error analysis
python src/evaluation/error_analysis.py
```

### 5. Statistical Analysis

```bash
# McNemar's test with Holm-Bonferroni correction + bootstrap CIs
python src/analysis/statistical_tests.py

# Feature importance & explainability
python src/analysis/explainability.py
```

### 6. Paper Generation

```bash
# Generate publication-ready figures
python src/analysis/generate_paper_figures.py

# Generate LaTeX tables
python src/analysis/generate_paper_tables.py
```

## Models

| Model               | Type           | Features                        | Parameters |
| ------------------- | -------------- | ------------------------------- | ---------- |
| Logistic Regression | Traditional ML | TF-IDF (10K vocab, uni+bigrams) | ~10K       |
| SVM (RBF kernel)    | Traditional ML | TF-IDF (10K vocab, uni+bigrams) | ~10K       |
| BiLSTM              | Deep Learning  | Word embeddings (dim=256)       | ~6.2M      |
| PhoBERT             | Transformer    | Subword tokens (phobert-base)   | ~135M      |

## Dataset

- **Total**: 5,655 Vietnamese news articles (after deduplication)
- **Classes**: Real News (62.9%), Fake News (37.1%)
- **Split**: 70% train / 15% validation / 15% test (stratified)
- **Preprocessing**: VnCoreNLP word segmentation, URL removal, text normalization

## Statistical Validation

- **McNemar's test** with Holm-Bonferroni correction for multiple comparisons
- **Bootstrap confidence intervals** (10,000 iterations)
- **Cohen's d** effect size analysis
- **5-fold cross-validation** with 3 random seeds for traditional ML models

## Citation

```bibtex
@article{fakenewsdetector2026,
  title={Vietnamese Fake News Detection: A Comparative Study of Machine Learning Approaches},
  year={2026}
}
```

## License

This project is for academic research purposes.
