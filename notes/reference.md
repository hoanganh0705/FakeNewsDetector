# GitHub References for Fake News Detection Project

## Project Context

- Task: Fake News Detection (Binary Classification)
- Dataset: ****\*\*****\_\_****\*\*****
- Goal: Compare classical ML, deep learning, and transformer-based models
- Models: Logistic Regression, SVM, BiLSTM, BERT / PhoBERT

---

## Model 1: Logistic Regression (Baseline)

### Reference Repository

- GitHub: https://github.com/nishitpatel01/Fake_News_Detection
- Model Type: Classical Machine Learning
- Feature Representation: TF-IDF

### Why This Repo

- Clean TF-IDF + Logistic Regression pipeline
- Beginner-friendly
- Matches baseline used in literature (e.g., Mathews & Preethi)

### What to Study

- TF-IDF construction
- Train/test split strategy
- Evaluation metrics (accuracy, F1)

### What I Will Reuse (Conceptually)

- TF-IDF + LR pipeline
- sklearn-based evaluation

---

## Model 2: Support Vector Machine (Strong ML Baseline)

### Reference Repository

- GitHub: https://github.com/abiek12/Fake-News-Detection-using-MachineLearning
- Model Type: Classical Machine Learning
- Feature Representation: TF-IDF

### Why This Repo

- Implements SVM for fake news detection
- Reflects strong baseline used in multiple papers
- Practical ML pipeline

### What to Study

- SVM configuration (kernel, parameters)
- Feature pipeline reuse
- Metric reporting

### What I Will Reuse (Conceptually)

- SVM + TF-IDF approach
- Same data split as LR

---

## Model 3: BiLSTM (Sequence-Based Deep Learning)

### Reference Repository

- GitHub: https://github.com/kanchanchy/Fake-News-Detection-BiLSTM
- Model Type: Deep Learning (Sequence Model)
- Feature Representation: Tokenized sequences + embeddings

### Why This Repo

- Pure BiLSTM implementation
- Clear tokenization and padding
- No transformer confusion

### What to Study

- Tokenizer usage
- Sequence padding
- BiLSTM architecture design

### What I Will Reuse (Conceptually)

- Tokenization → padding → BiLSTM flow
- Embedding + BiLSTM structure

---

## Model 4: BERT / PhoBERT (Transformer – State of the Art)

### Reference Repository

- GitHub (BERT): https://github.com/wutonytt/Fake-News-Detection
- Model Type: Transformer
- Feature Representation: Contextual embeddings (BERT tokenizer)

> If dataset is Vietnamese → replace BERT with PhoBERT

### Why This Repo

- Uses Hugging Face `transformers`
- Proper fine-tuning workflow
- Matches modern NLP research standards

### What to Study

- Tokenizer usage
- Fine-tuning loop
- Batch size & max sequence length handling

### What I Will Reuse (Conceptually)

- Hugging Face pipeline
- Fine-tuning strategy

---

## Cross-Model Notes

### Common Rules

- Same dataset
- Same train/validation/test split
- Same evaluation metrics

### Cautions

- Do NOT reuse TF-IDF for BERT
- Do NOT remove stopwords for BERT / PhoBERT
- Avoid data leakage when splitting

### Research Insights

- ***
- ***

---

## Paper Writing Reminders

- Cite GitHub repos as implementation references (not contributions)
- Emphasize independent implementation
- Highlight BiLSTM as added novelty vs prior work
