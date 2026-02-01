# Vietnamese Fake News Detection: A Comparative Study of Machine Learning Approaches

## Abstract

This study presents a comprehensive comparison of machine learning approaches for Vietnamese fake news detection. We evaluate four models representing different paradigms: Logistic Regression and Support Vector Machine (traditional machine learning with TF-IDF features), Bidirectional LSTM (deep learning with word embeddings), and PhoBERT (transformer-based pre-trained language model). Our experiments on a dataset of 5,655 Vietnamese news articles demonstrate that PhoBERT achieves the best performance with 90.58% accuracy and 0.896 F1-score, significantly outperforming traditional approaches (p < 0.05). The results highlight the effectiveness of pre-trained language models for Vietnamese NLP tasks while providing insights into the trade-offs between model complexity and performance.

**Keywords:** Fake news detection, Vietnamese NLP, Machine learning, Deep learning, PhoBERT, Text classification

---

## 1. Introduction

The proliferation of fake news on social media platforms has become a significant concern for society, particularly in Vietnamese-speaking communities. Misinformation can influence public opinion, affect democratic processes, and even endanger public health. Automatic fake news detection systems are essential tools for combating this challenge.

While extensive research has been conducted on fake news detection for English text, the Vietnamese language presents unique challenges due to its linguistic characteristics, including the use of diacritics, compound words, and the absence of word boundaries. This study addresses the gap by conducting a systematic comparison of multiple machine learning approaches for Vietnamese fake news detection.

### Research Objectives

1. Compare the effectiveness of traditional machine learning, deep learning, and transformer-based approaches for Vietnamese fake news detection
2. Analyze the performance characteristics of each model paradigm
3. Provide recommendations for practitioners building Vietnamese fake news detection systems

---

## 2. Related Work

### 2.1 Fake News Detection

Previous research on fake news detection has employed various approaches:

- **Linguistic features**: Analysis of writing style, sentiment, and lexical patterns
- **Network-based methods**: Analyzing the spread patterns on social networks
- **Knowledge-based approaches**: Fact-checking against knowledge bases

### 2.2 Vietnamese NLP

Vietnamese natural language processing has seen significant advances with:

- **VnCoreNLP**: A toolkit for word segmentation, POS tagging, and dependency parsing
- **PhoBERT**: The first public large-scale monolingual language model for Vietnamese
- **ViT5**: Vietnamese T5 models for sequence-to-sequence tasks

---

## 3. Methodology

### 3.1 Dataset

We compiled a dataset of Vietnamese news articles labeled as real or fake. After preprocessing and removing duplicates:

| Split      | Total | Real News     | Fake News     | Avg. Length |
| ---------- | ----- | ------------- | ------------- | ----------- |
| Training   | 3,957 | 2,490 (62.9%) | 1,467 (37.1%) | ~150 words  |
| Validation | 849   | 534 (62.9%)   | 315 (37.1%)   | ~150 words  |
| Test       | 849   | 534 (62.9%)   | 315 (37.1%)   | ~150 words  |

### 3.2 Text Preprocessing

1. **Word Segmentation**: Using VnCoreNLP to segment Vietnamese text into tokens
2. **Text Cleaning**: Removing URLs, special characters, and normalizing whitespace
3. **Lowercasing**: Converting all text to lowercase

### 3.3 Feature Engineering

#### TF-IDF Features (for LR and SVM)

- Vocabulary size: 10,000 terms
- N-gram range: (1, 2)
- Sublinear TF scaling enabled

#### Word Embeddings (for BiLSTM)

- Vocabulary built from training data (22,852 words)
- Embedding dimension: 128
- Maximum sequence length: 200

#### PhoBERT Tokenization

- Using `vinai/phobert-base` tokenizer
- Maximum sequence length: 256 tokens

### 3.4 Models

#### Logistic Regression

- L2 regularization with C=10
- LBFGS solver
- Class weights for handling imbalance

#### Support Vector Machine

- RBF kernel with C=10, gamma=scale
- Probability estimates enabled
- Class weights for handling imbalance

#### BiLSTM

- 2-layer bidirectional LSTM
- Hidden dimension: 128
- Dropout: 0.3
- Adam optimizer with learning rate 0.001
- Early stopping with patience 3

#### PhoBERT

- Pre-trained `vinai/phobert-base` model
- Fine-tuned for classification
- Learning rate: 2e-5
- 5 training epochs

---

## 4. Results

### 4.1 Overall Performance

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **PhoBERT**         | **0.9058** | **0.9114** | **0.8860** | **0.8961** | **0.9578** |
| SVM                 | 0.8763     | 0.8753     | 0.8568     | 0.8644     | 0.9484     |
| Logistic Regression | 0.8728     | 0.8627     | 0.8663     | 0.8644     | 0.9419     |
| BiLSTM              | 0.8728     | 0.8618     | 0.8696     | 0.8652     | 0.9385     |

### 4.2 Statistical Significance

McNemar's test results show that PhoBERT's improvement over other models is statistically significant (p < 0.05):

| Comparison        | χ²   | p-value | Significant |
| ----------------- | ---- | ------- | ----------- |
| PhoBERT vs LR     | 6.51 | 0.011   | Yes\*       |
| PhoBERT vs SVM    | 6.33 | 0.012   | Yes\*       |
| PhoBERT vs BiLSTM | 6.39 | 0.011   | Yes\*       |
| LR vs SVM         | 0.09 | 0.771   | No          |
| LR vs BiLSTM      | 0.01 | 0.918   | No          |
| SVM vs BiLSTM     | 0.04 | 0.842   | No          |

\*Significant at α = 0.05

### 4.3 Bootstrap Confidence Intervals (95%)

| Model               | Accuracy CI      | F1-Score CI      |
| ------------------- | ---------------- | ---------------- |
| PhoBERT             | [0.8846, 0.9246] | [0.8723, 0.9177] |
| SVM                 | [0.8539, 0.8964] | [0.8397, 0.8862] |
| Logistic Regression | [0.8504, 0.8940] | [0.8401, 0.8868] |
| BiLSTM              | [0.8481, 0.8952] | [0.8385, 0.8878] |

### 4.4 Per-Class Performance

| Model   | Class | Precision | Recall | F1-Score |
| ------- | ----- | --------- | ------ | -------- |
| PhoBERT | Real  | 0.8955    | 0.9625 | 0.9278   |
| PhoBERT | Fake  | 0.9273    | 0.8095 | 0.8644   |
| SVM     | Real  | 0.8783    | 0.9326 | 0.9046   |
| SVM     | Fake  | 0.8723    | 0.7810 | 0.8241   |
| LR      | Real  | 0.9049    | 0.8914 | 0.8981   |
| LR      | Fake  | 0.8204    | 0.8413 | 0.8307   |
| BiLSTM  | Real  | 0.9128    | 0.8820 | 0.8971   |
| BiLSTM  | Fake  | 0.8108    | 0.8571 | 0.8333   |

### 4.5 Error Analysis

- **Total hard examples**: 29 samples misclassified by all models
- **Common error patterns**: Shorter texts are harder to classify correctly
- **False positive rate**: PhoBERT has the lowest (3.7%)
- **False negative rate**: BiLSTM has the lowest (14.3%)

---

## 5. Discussion

### 5.1 Key Findings

1. **PhoBERT achieves superior performance**: With 90.58% accuracy and 0.896 F1-score, PhoBERT significantly outperforms all other models. This demonstrates the effectiveness of pre-trained language models for Vietnamese NLP tasks.

2. **Traditional ML remains competitive**: Logistic Regression and SVM achieve comparable performance to BiLSTM (~87% accuracy), suggesting that simple models with well-engineered features are still viable options.

3. **Statistical significance**: The improvement of PhoBERT over other models is statistically significant (p < 0.05), confirming that the performance gains are not due to random chance.

4. **Trade-off considerations**:
   - **Training time**: LR (3s) < BiLSTM (18s) < SVM (119s) < PhoBERT (311s)
   - **Model size**: Traditional ML (~10K params) << BiLSTM (~500K) << PhoBERT (~135M)
   - **Inference speed**: Traditional ML > BiLSTM > PhoBERT

### 5.2 Practical Recommendations

- **For high-accuracy requirements**: Use PhoBERT when computational resources are available
- **For resource-constrained environments**: SVM or Logistic Regression provide good accuracy with minimal resources
- **For real-time applications**: Traditional ML models offer the fastest inference

### 5.3 Limitations

1. **Dataset size**: The dataset contains ~5,600 samples, which may not capture all patterns
2. **Domain coverage**: The data may not represent all types of fake news
3. **Temporal aspects**: News patterns evolve over time

---

## 6. Conclusion

This study presents a comprehensive comparison of machine learning approaches for Vietnamese fake news detection. Our experiments demonstrate that PhoBERT, a pre-trained transformer model, achieves the best performance with 90.58% accuracy, significantly outperforming traditional machine learning and deep learning approaches. The 3.7% improvement over baseline models is statistically significant (p < 0.05).

### Future Work

1. **Multimodal detection**: Incorporating images and metadata
2. **Explainability**: Adding interpretability to understand model decisions
3. **Cross-domain evaluation**: Testing on different news sources
4. **Larger datasets**: Collecting more diverse training data

---

## References

1. Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP 2020.

2. Vu, X. S., et al. (2019). VnCoreNLP: A Vietnamese Natural Language Processing Toolkit. NAACL 2018.

3. Zhou, X., & Zafarani, R. (2020). A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities. ACM Computing Surveys.

4. Shu, K., et al. (2017). Fake News Detection on Social Media: A Data Mining Perspective. ACM SIGKDD Explorations Newsletter.

---

## Appendix

### A. Experimental Setup

- **Hardware**: NVIDIA GPU with CUDA support
- **Software**: Python 3.14, PyTorch, scikit-learn, Transformers
- **Random seed**: 42 for reproducibility

### B. Hyperparameter Search

Grid search was performed for traditional ML models:

- **Logistic Regression**: C ∈ {0.1, 1, 10}
- **SVM**: C ∈ {0.1, 1, 10}, kernel ∈ {linear, rbf}

### C. Code Availability

The code and trained models are available at: [Repository URL]
