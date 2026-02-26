# Vietnamese Fake News Detection: A Comparative Study of Machine Learning Approaches

A research project comparing machine learning approaches for Vietnamese fake news detection, evaluating four models across different paradigms: traditional ML (Logistic Regression, SVM), deep learning (BiLSTM), and transformer-based (PhoBERT).

## Key Results

| Model               | Accuracy   | F1-Score  |
| ------------------- | ---------- | --------- |
| **PhoBERT**         | **92.25%** | **0.920** |
| SVM                 | 87.31%     | 0.867     |
| BiLSTM              | 86.05%     | 0.855     |
| Logistic Regression | 85.61%     | 0.852     |

PhoBERT significantly outperforms all baselines (McNemar's test with Holm-Bonferroni correction, p ≈ 0.0000).

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
| Logistic Regression | Traditional ML | TF-IDF (10K vocab, uni+bigrams) | ~10K       |
| SVM (RBF kernel)    | Traditional ML | TF-IDF (10K vocab, uni+bigrams) | ~10K       |
| BiLSTM              | Deep Learning  | Word embeddings (dim=256)       | ~6.2M      |
| PhoBERT             | Transformer    | Subword tokens (phobert-base)   | ~135M      |

## Dataset

- **Total**: 10,097 Vietnamese news articles (9,031 after cleaning)
- **Classes**: Real (58.6%) / Fake (41.4%)
- **Split**: 70% train / 15% validation / 15% test (stratified)
- **Preprocessing**: VnCoreNLP word segmentation (RDRSegmenter), URL removal, text normalization

## Statistical Validation

- **McNemar's test** with Holm-Bonferroni correction for multiple comparisons
- **Bootstrap confidence intervals** (10,000 iterations)
- **Cohen's d** effect size analysis
- **5-fold cross-validation** with 3 random seeds for traditional ML models

## License

This project is for academic research purposes.
