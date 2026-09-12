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

## Paper source and LaTeX lists

The main paper source is `paper/main.tex`. The compiled output is `paper/main.pdf`.

To build the PDF (4-pass pipeline: pdflatex → bibtex → pdflatex → pdflatex):

```bash
cd paper
./build.sh          # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
./build.sh clean    # xóa các file trung gian (.aux, .toc, .bbl, …)
```

Requires a TeX Live installation (tested with TeX Live 2025). The build script runs the
standard `pdflatex` + `bibtex` cycle so that citations, cross-references and the table of
contents settle correctly.

When adding or editing a figure/table, keep two caption versions:

```latex
\caption[Short caption for the list]{Full caption shown beside the figure/table}
```

The short version is used automatically by `listoffigures` and `listoftables`; the full version remains visible in the report body. This is especially useful for long analytical captions.

### File layout

The paper is a single `main.tex` (~2,600 lines) organised as follows:

| Lines (approx.) | Section                                                |
| --------------- | ------------------------------------------------------ |
| 1 – 100         | Document class, packages, hyperref setup                |
| 100 – 470       | Centralised `\newcommand`s for every dataset/metric    |
| 470 – 690       | Title page, ToC, lists of tables/figures, abbreviations |
| 690 – 1,200     | Chapter 1 — Cơ sở lý thuyết                              |
| 1,200 – 1,400   | Chapter 2 — Bộ dữ liệu và tiền xử lý                     |
| 1,400 – 1,660   | Chapter 3 — Xác định tin giả                              |
| 1,660 – 2,545   | Chapter 4 — Kết quả thực nghiệm                           |
| 2,545 – 2,590   | Kết luận, Tài liệu tham khảo                              |

Numerical results live in the `\newcommand` block near the top of `main.tex`. Editing a
value there propagates everywhere — table `\input`s, body text, captions.


A research/demo interface that loads the existing trained models and runs
inference on a user-pasted Vietnamese article. **No model is retrained.**
If a checkpoint or feature file is missing, the demo shows a clear error
explaining which artefact is required.

## Streamlit Demo

```bash
cd FakeNewsDetector
source .venv/bin/activate
pip install -r requirements.txt        # adds streamlit
streamlit run app.py
```

Then open the local URL Streamlit prints (default: <http://localhost:8501>).

The demo uses the same label convention as the research pipeline
(0 = Real, 1 = Fake) and surfaces the published benchmark scores
separately from the model's live prediction.

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
