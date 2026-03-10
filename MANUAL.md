# Vietnamese Fake News Detection — Run Manual

Complete guide to run the full pipeline from scratch.

---

## Prerequisites

| Requirement     | Version | Notes                                |
| --------------- | ------- | ------------------------------------ |
| Python          | 3.10+   | Tested on 3.14.2                     |
| Java JRE        | 11+     | For py_vncorenlp word segmentation   |
| CUDA (optional) | 11.8+   | GPU recommended for BiLSTM & PhoBERT |

---

## 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/hoanganh0705/FakeNewsDetector.git
cd FakeNewsDetector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install the project in editable mode (preferred — also installs the `fakenews` CLI)
# The CFLAGS/CXXFLAGS fix a missing <cstdint> include in fasttext-wheel on Python 3.14+
CFLAGS="-include cstdint" CXXFLAGS="-include cstdint" pip install -e ".[dev]"

# Or, install from requirements.txt only (no CLI command)
# CFLAGS="-include cstdint" CXXFLAGS="-include cstdint" pip install -r requirements.txt

# Set JAVA_HOME (needed for py_vncorenlp)
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk   # adjust to your Java path
# To find your Java path: dirname $(dirname $(readlink -f $(which java)))
```

---

## 2. Prepare Data

Place your raw data at `data/raw/raw.csv` with columns: `id`, `text`, `date`, `label` (0=Real, 1=Fake).

### Download pretrained FastText embeddings (for BiLSTM)

The BiLSTM model uses pretrained Vietnamese FastText word vectors (`cc.vi.300.bin`).

```bash
mkdir -p data/fasttext
# Download (~4.5 GB compressed, ~7 GB unzipped)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz -O data/fasttext/cc.vi.300.bin.gz
gunzip data/fasttext/cc.vi.300.bin.gz
```

The config (`cfg.BILSTM.fasttext_path`) already points to `data/fasttext/cc.vi.300.bin`.
If you place the file elsewhere, update `fasttext_path` in `config.py`.

---

## 3. Run the Pipeline

All commands are run from the project root with the venv activated.

> **Tip:** If you installed with `pip install -e .` you can use the `fakenews` CLI
> instead of `python src/…` — see [CLI Reference](#cli-reference) below.

### Step 1: Word Segmentation

Segments Vietnamese text using VnCoreNLP's RDRSegmenter (same tokenizer PhoBERT was trained with).

```bash
python src/preprocessing/word_segmentation.py
# or: fakenews preprocess
```

**Input:** `data/raw/raw.csv`
**Output:** `data/processed/segmented.csv`
**Time:** ~30 seconds for 10K samples

### Step 2: Train/Val/Test Split

Stratified 70/15/15 split with duplicate removal and data leakage check.

```bash
python src/preprocessing/split_data.py
# or: fakenews split
```

**Output:** `data/splits/{train,val,test}.csv`

### Step 3: Feature Extraction

Extracts TF-IDF (for LR & SVM), word embeddings (for BiLSTM), and PhoBERT tokens (for transformer).

```bash
python src/features/extract_all_features.py
# or: fakenews features
```

**Output:** `data/features/{tfidf,embedding,phobert}/`
**Time:** ~10 seconds

### Step 4: Train All Models

Trains Logistic Regression, SVM, BiLSTM, and PhoBERT sequentially.

```bash
python src/training/train_all.py
# or: fakenews train              # trains all models
# or: fakenews train lr svm       # train specific models only
```

**Output:** `experiments/{lr,svm,bilstm,bert}/` (models + `metrics.json` + `predictions.pkl`)
**Time:** ~15–20 minutes (with GPU), ~1+ hour (CPU only)

### Step 5: Evaluate All Models

Generates comparison tables, confusion matrices, ROC curves, and training history plots.

```bash
python src/evaluation/evaluate_all.py
# or: fakenews evaluate
```

**Output:**

- `results/tables/model_comparison.csv`, `evaluation_summary.json`, `per_class_metrics.csv`
- `results/figures/model_comparison.png`, `confusion_matrices.png`, `roc_curves.png`, `training_history.png`
- `experiments/*/predictions.pkl` (if not already saved by training)

### Step 6: Error Analysis

Analyzes errors per model, identifies hard examples all models get wrong.

```bash
python src/evaluation/error_analysis.py
```

**Output:**

- `results/tables/hard_examples.csv`, `per_id_confidence.csv`
- `results/figures/error_analysis/`

### Step 7: Statistical Significance Tests

McNemar's test with Holm-Bonferroni correction, bootstrap CIs (10K iterations), Cohen's d.

```bash
python src/analysis/statistical_tests.py
```

**Output:** `results/tables/mcnemar_test.csv`, `confidence_intervals.csv`, `effect_sizes.csv`, `statistical_analysis_summary.json`

### Step 8: Cross-Validation

5-fold stratified CV with 3 random seeds (15 total folds) for LR and SVM.

```bash
python src/evaluation/cross_validation.py
```

**Output:** `results/tables/cross_validation_summary.csv`, `cross_validation_results.json`

### Step 9: Ablation Study

Evaluates impact of vocabulary size, n-grams, word segmentation, regularization, and TF scaling.

```bash
python src/evaluation/ablation_study.py
```

**Output:** `results/tables/ablation_study.json`, `ablation_study.tex`

### Step 10: Explainability Analysis

LR feature importance (top predictive words) and error taxonomy by text length / confidence.

```bash
python src/analysis/explainability.py
```

**Output:**

- `results/figures/explainability/feature_importance.png`, `error_taxonomy.png`
- `paper/figures/feature_importance.pdf`, `error_taxonomy.pdf`

### Step 11: Generate Paper Figures

Publication-quality figures (300 DPI, serif fonts, PDF + PNG).

```bash
python src/analysis/generate_paper_figures.py
```

**Output:** `paper/figures/fig{0-6}_*.{png,pdf}`

### Step 12: Generate Paper Tables

LaTeX-formatted tables ready for `\input{}` in the paper.

```bash
python src/analysis/generate_paper_tables.py
```

**Output:** `paper/tables/table{1-5}_*.tex`

---

## Quick Run (All Steps)

### Using the CLI (recommended)

```bash
source .venv/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk

# Steps 1–5 in one command
fakenews run

# Then run the remaining analysis steps
python src/evaluation/error_analysis.py
python src/analysis/statistical_tests.py
python src/evaluation/cross_validation.py
python src/evaluation/ablation_study.py
python src/analysis/explainability.py
python src/analysis/generate_paper_figures.py
python src/analysis/generate_paper_tables.py
```

### Using individual scripts

```bash
source .venv/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk

# Full pipeline
python src/preprocessing/word_segmentation.py
python src/preprocessing/split_data.py
python src/features/extract_all_features.py
python src/training/train_all.py
python src/evaluation/evaluate_all.py
python src/evaluation/error_analysis.py
python src/analysis/statistical_tests.py
python src/evaluation/cross_validation.py
python src/evaluation/ablation_study.py
python src/analysis/explainability.py
python src/analysis/generate_paper_figures.py
python src/analysis/generate_paper_tables.py
```

Total runtime: ~20–25 minutes (with GPU).

---

## CLI Reference

After `pip install -e .`, the `fakenews` command is available:

```
fakenews preprocess          # Step 1: word segmentation
fakenews split               # Step 2: train/val/test split
fakenews features            # Step 3: extract all features
fakenews train [MODEL ...]   # Step 4: train models (lr, svm, bilstm, phobert, all)
fakenews evaluate            # Step 5: full evaluation suite
fakenews run                 # Steps 1–5 end-to-end
```

---

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_smoke.py -v
```

---

## Configuration

All hyperparameters and paths are centralised in `config.py`. Key settings:

| Setting                     | Default | Description                        |
| --------------------------- | ------- | ---------------------------------- |
| `cfg.RANDOM_STATE`          | 42      | Global random seed                 |
| `cfg.PHOBERT.learning_rate` | 2e-5    | PhoBERT fine-tuning LR             |
| `cfg.PHOBERT.dropout`       | 0.1     | PhoBERT classifier dropout         |
| `cfg.PHOBERT.batch_size`    | 16      | PhoBERT batch size                 |
| `cfg.BILSTM.batch_size`     | 64      | BiLSTM batch size                  |
| `cfg.LR.C`                  | 1.0     | Logistic Regression regularisation |
| `cfg.SVM.C`                 | 1.0     | SVM regularisation                 |

To adjust GPU memory usage, change `cfg.PHOBERT.batch_size` or `cfg.BILSTM.batch_size` in `config.py` instead of editing training scripts directly.

---

## Project Structure (Outputs)

```
FakeNewsDetector/
├── data/
│   ├── raw/raw.csv                    # Input data
│   ├── processed/segmented.csv        # After word segmentation
│   ├── splits/{train,val,test}.csv    # Data splits
│   └── features/                      # Extracted features
│       ├── tfidf/
│       ├── embedding/
│       └── phobert/
├── experiments/                       # Trained models + metrics
│   ├── lr/
│   ├── svm/
│   ├── bilstm/
│   └── bert/
├── results/
│   ├── figures/                       # Evaluation visualizations
│   └── tables/                        # CSV & LaTeX results
└── paper/
    ├── figures/                       # Publication figures
    └── tables/                        # Publication tables
```

---

## Troubleshooting

| Problem                    | Solution                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `py_vncorenlp unavailable` | Install: `pip install py_vncorenlp` and set `JAVA_HOME`                               |
| `Unable to find javac`     | Set `export JAVA_HOME=/path/to/jvm` (JRE is sufficient with JAVA_HOME set)            |
| CUDA out of memory         | Reduce `cfg.PHOBERT.batch_size` or `cfg.BILSTM.batch_size` in `config.py`             |
| `No module named 'src'`    | Install with `pip install -e .` or run from project root                              |
| Slow PhoBERT training      | Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"` |

---

## Notes

- **Random seed:** 42 (fixed throughout for reproducibility)
- **Word segmenter:** py_vncorenlp (RDRSegmenter) — the same tokenizer PhoBERT was trained with. Using a different segmenter (e.g., underthesea) will degrade PhoBERT performance.
- **Linting:** The project uses [Ruff](https://docs.astral.sh/ruff/) — config is in `pyproject.toml`. Run `ruff check .` to lint or `ruff format .` to auto-format.
- **If data changes:** Delete everything under `data/processed/`, `data/splits/`, `data/features/`, `experiments/`, `results/`, and `paper/figures/`, `paper/tables/`, then re-run from Step 1.
