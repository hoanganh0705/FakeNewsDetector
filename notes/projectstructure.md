# FakeNewsDetector - Project Structure

A Machine Learning project for detecting fake news using multiple model architectures.

## 📁 Core Code Folders

### `src/` - Source code directory

Contains the main application code organized by functionality:

- **`models/`** - Model architecture definitions (BERT, BiLSTM, Logistic Regression, SVM) _(currently empty)_
- **`preprocessing/`** - Data cleaning, text preprocessing, and feature extraction code _(currently empty)_
- **`training/`** - Training scripts and pipelines for your models _(currently empty)_
- **`evaluation/`** - Model evaluation metrics and testing code _(currently empty)_

## 📊 Data Folders

### `data/` - All datasets for the project

- **`raw/`** - Original, unprocessed datasets
- **`processed/`** - Cleaned and preprocessed data ready for model training
- **`splits/`** - Train/validation/test dataset splits

## 🧪 Experiment Folders

### `experiments/` - Experiment configurations and results

Organized by model type:

- **`bert/`** - BERT transformer model experiments
- **`bilstm/`** - Bidirectional LSTM model experiments
- **`lr/`** - Logistic Regression baseline experiments
- **`svm/`** - Support Vector Machine baseline experiments

## 📈 Results Folders

### `results/` - Output from experiments

- **`figures/`** - Visualizations, charts, and plots
- **`logs/`** - Training logs and experiment tracking
- **`tables/`** - Performance metrics and comparison tables

## 📝 Documentation Folders

- **`paper/`** - Research paper drafts and LaTeX files _(currently empty)_
- **`notes/`** - Project notes and documentation

## 📄 Root Files

- **`requirements.txt`** - Python dependencies (numpy, pandas, scikit-learn, PyTorch, transformers, etc.)
- **`README.md`** - Project documentation _(currently empty)_
- **`note.md`** - Quick notes file

## Project Status

The project structure follows a typical ML research project layout. The `src/` folders are currently empty and ready for implementation code.
