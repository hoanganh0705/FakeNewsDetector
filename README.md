# Vietnamese Fake News Detection: A Comparative Study of Machine Learning Approaches

A research project comparing machine learning approaches for Vietnamese fake news detection, evaluating four models across different paradigms: traditional ML (Logistic Regression, SVM), deep learning (BiLSTM), and transformer-based (PhoBERT).

## Key Results

| Model               | Accuracy   | F1-Score  |
| ------------------- | ---------- | --------- |
| **PhoBERT**         | **90.07%** | **0.899** |
| SVM                 | 84.34%     | 0.841     |
| BiLSTM              | 82.52%     | 0.823     |
| Logistic Regression | 83.48%     | 0.833     |

PhoBERT outperforms all baselines on the test set (verified via McNemar's test, p < 0.001).

## Project Structure

```
FakeNewsDetector/
├── data/
│   ├── raw/raw.csv                     # Original dataset
│   ├── processed/segmented.csv         # Word-segmented text
│   ├── splits/                         # Train/Val/Test splits (70/15/15)
│   └── features/                       # Extracted features
├── src/
│   ├── preprocessing/                  # Text cleaning & word segmentation
│   ├── features/                       # Feature extraction (TF-IDF, embeddings, PhoBERT)
│   ├── training/                       # Model training scripts
│   ├── evaluation/                     # Metrics, error analysis, cross-validation, ablation
│   └── analysis/                       # Statistical tests, explainability, paper generation
├── experiments/                        # Trained models & metrics
├── results/                            # Figures, tables, evaluation outputs
└── paper/                              # LaTeX paper, figures, tables
```

## Quick Start

```bash
git clone https://github.com/hoanganh0705/FakeNewsDetector.git
cd FakeNewsDetector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk  # adjust to your Java path
```

See [MANUAL.md](MANUAL.md) for the full step-by-step guide.

## Models

| Model               | Type           | Features                        | Parameters |
| ------------------- | -------------- | ------------------------------- | ---------- |
| Logistic Regression | Traditional ML | TF-IDF (27.6K vocab, uni+bigrams) | ~27.6K     |
| SVM (LinearSVC)    | Traditional ML | TF-IDF (27.6K vocab, uni+bigrams) | ~27.6K     |
| BiLSTM              | Deep Learning  | Word embeddings (dim=300, hidden=128) | ~7.5M     |
| PhoBERT             | Transformer    | Subword tokens (phobert-base, 256 len) | ~134M    |

## Dataset

- **Raw**: 15,789 Vietnamese news articles
- **After cleaning**: 13,958 articles (7,764 Real / 6,194 Fake)
- **Classes**: Real (55.6%) / Fake (44.4%)
- **Split**: 70% train (9,770) / 15% validation (2,094) / 15% test (2,094), stratified
- **Preprocessing**: VnCoreNLP word segmentation (RDRSegmenter), URL removal, text normalization

## Statistical Validation

- **McNemar's test** with Holm-Bonferroni correction for multiple comparisons
- **Bootstrap confidence intervals** (10,000 iterations)
- **Cohen's d** effect size analysis
- **5-fold cross-validation** with 3 random seeds for traditional ML models

## License

This project is for academic research purposes.
