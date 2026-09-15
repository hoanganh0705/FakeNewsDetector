# 🔬 FULL PROJECT AUDIT — Vietnamese Fake News Detector

**Scope of this audit**: Vietnamese-only fake news detection.
**Date of audit**: 2026-09-15.
**Repository audited**: `FakeNewsDetector` (commit on disk, 2026-09-15).

---

## Table of Contents

1. [Project Map](#technical-map)
2. [Phase 1 — Full Project Audit](#phase-1--full-project-audit)
3. [Phase 2 — Current Project Level](#phase-2--current-project-level)
4. [Phase 3 — Research Gaps](#phase-3--research-gaps)
5. [Phase 4 — Eight+ Thesis Directions (Vietnamese-only)](#phase-4--eight-thesis-directions-vietnamese-only)
6. [Phase 5 — Research Axes Summary](#phase-5--research-axes-summary)
7. [Phase 6 — Academic Contribution Analysis](#phase-6--academic-contribution-analysis)
8. [Phase 7 — Experimental Design (Top 3 Directions)](#phase-7--experimental-design-top-3-directions)
9. [Phase 8 — Ranking](#phase-8--ranking)
10. [Phase 9 — Proposed Final Thesis](#phase-9--proposed-final-thesis)
11. [Phase 10 — Implementation Roadmap](#phase-10--implementation-roadmap)
12. [🔥 Final Advisor Recommendation](#-final-advisor-recommendation)

---

## Technical Map

```
[User]
 ↓
[app.py — Streamlit UI] (inference only, no retraining)
   ↓
[src/cli.py] fakenews preprocess|split|features|train|evaluate|calibration|explain|recalibrate|hard-cases|distill|run
   ↓
[Preprocessing]              [Features]                [Models]
 src/preprocessing/           src/features/             src/models/
 • text_preprocessor          • tfidf_features (LR/SVM) • phobert_model
 • word_segmentation • embedding_features       • bilstm_model
   (py_vncorenlp /            (BiLSTM) • student_model
 underthesea)              • phobert_features
 • split_data                   (PhoBERT)
   (70/15/15 stratified)

[Training]                    [Evaluation]              [Analysis]
 src/training/                src/evaluation/           src/analysis/
 • runner (shared save)       • metrics • explainability (LR feature importance, error cat.)
 • train_lr, train_svm        • evaluate_all            • explainability_runner (SHAP/IG/attention)
 • train_bilstm, train_phobert• cross_validation        • lr_svm_shap • train_student (KD)         • calibration_analysis • bilstm_attribution
 • train_all • post_hoc_calibration    • phobert_attribution
 • distillation_evaluation    • error_analysis • method_agreement (cross-model)
 • phase0_lr_svm_logits • hard_cases              • statistical_tests (McNemar, bootstrap, Cohen's d)
 • reproduce_predictions • ablation_study • generate_paper_figures/tables
```

---

## PHASE 1 — FULL PROJECT AUDIT

### A. Problem Definition

| Aspect | Current State (verified) |
|---|---|
| **Task** | Binary text classification — Real (0) vs Fake (1) |
| **Language** | Vietnamese only |
| **Input** | Pre-tokenized Vietnamese news article (`text` column) |
| **Output** | Class label + P(Fake) |
| **Definition of "fake"** | Implicit; dataset-supplied. The README admits: **"Dataset source/licence: UNKNOWN — see the thesis for source attribution."** |
| **Multiclass?** | No — strictly binary. |
| **Misinformation/credibility/stance?** | None — only veracity-style fake news. |

**Critical caveat**: The README itself flags that the dataset source/licence is **UNKNOWN**. This is a major academic gap — for a thesis, you must trace and document dataset provenance. The dataset contains Vietnamese social-media posts (some with `<URL>`, `#hashtags`, Twitter handles, emojis) and formal news — suggesting it is **not a single homogeneous source** but a mix.

**Misclassification notes from manual inspection of `raw.csv`**:

- Some "Fake" labels look like opinion/anti-government essays rather than misinformation (e.g. rows 11, 14, 17, 44, 50 — long political essays).
- Some "Real" labels look like substantive reporting.
- Many samples contain `<URL>` placeholders, hashtags, and emojis — strong style cues the model may exploit (potential **stylometric shortcut** rather than learning content credibility).

### B. Dataset

| Property | Value | Verified |
|---|---|---|
| File | `data/raw/raw.csv` | ✅ |
| Rows (raw) | **15,789** | ✅ (`wc -l` + pandas) |
| Columns | `id, text, date, label` | ✅ |
| Class distribution | Real 8,355 (52.9%) / Fake 7,434 (47.1%) | ✅ |
| Median text length | 110 chars (~22 words) | ✅ |
| 25% quantile | 73 chars (very short) | ✅ |
| 75% quantile | 287 chars | ✅ |
| Max length | 32,832 chars (outlier) | ✅ |
| Min length | 4 chars | ✅ |
| Date range | 2015-04-02 → 2017-12-27 (532 valid dates) | ✅ |
| **Date missing rate** | **15,257 / 15,789 ≈96.6%** of `date` values are unparseable / missing | ✅ |
| Duplicates (text) | **886 exact duplicates** | ✅ |
| Language | Vietnamese only | ✅ (visual inspection) |
| Dataset source/licence | **UNKNOWN** (self-acknowledged in README) | ✅ |

**Major dataset limitations (verified):**

1. **Provenance unknown** — cannot be cited rigorously in a thesis. Some samples show clear hallmarks of social-media scraping (Twitter handles, `pic.twitter.com` URLs, hashtags) while others are formal news prose. This heterogeneity is *itself* a research question.

2. **~5.6% exact text duplicates (886/15,789)**. The split script does dedup (verified in `text_preprocessor.py::clean_dataset`), but if the same source article appears with minor URL/emoji variations, near-duplicates will survive. The README reports **13,958 articles after cleaning** — consistent with removing the 886 dups plus very-short records.

3. **96.6% missing dates** — temporal analysis is statistically impossible without recovering dates. The model also cannot learn any temporal signal because the test set is randomised.

4. **Stratified split is random** (single seed = 42, no time-based holdout). **No concept-drift / temporal-generalization evaluation exists.**

5. **Class balance is mild but not extreme** — no augmentation, no class-weighting tricks beyond `class_weight="balanced"` for LR/SVM only (BiLSTM and PhoBERT use balanced class weights via `compute_balanced_class_weights`).

6. **Stylometric artefacts**: `<URL>` placeholders, `#hashtags`, all-caps headlines (rows 4, 18, 21, 23, 28, 33, 51, 75, 79), emoji, `pic.twitter.com` — the model can achieve high accuracy by exploiting these surface features rather than semantic credibility. This is **a real research risk**.

### C. Preprocessing

| Step | Implementation | Verdict |
|---|---|---|
| Tokenization / segmentation | `py_vncorenlp` (RDRSegmenter) → fallback `underthesea`. Used **only for TF-IDF and BiLSTM** — PhoBERT uses its own BPE tokenizer | ✅ Correct: matches PhoBERT's pretokenizer |
| Cleaning | `clean_text`: lowercase, strip markdown URLs (`[...](...)`), bare URLs, `<URL>` placeholders, emails, HTML, "non-word" characters | ✅ Reasonable, but the regex `[^\w\s\u00C0-\u024F\u1E00-\u1EFF]` keeps diacritics (good) but also strips punctuation that may be informative (e.g. `!!!`, `???`, all-caps style cues) |
| Deduplication | `drop_duplicates(subset='text')` | ✅ |
| Length filter | `min_word_count=10` (config) | ✅ |
| Date normalisation | `pd.to_datetime(...).dt.strftime('%Y-%m-%d')` | ✅ but 96.6% are NaT → silently NaN |
| Stemming/lemmatization | **None** — relies on PhoBERT/BiLSTM to learn morphology | Acceptable for deep models |
| Stopword removal | **None** for any model | Acceptable (transformers benefit from stopwords) |
| Lowercasing | TF-IDF only; BiLSTM/PhoBERT keep case (good — case carries credibility cues) | ✅ |
| Augmentation | **None** | ⚠️ Gap for thesis |
| Vietnamese-specific handling | Word segmentation is correct | ✅ |
| Multilingual handling | N/A (Vietnamese only) | — |

**One subtle issue**: in `app.py::predict_bilstm`, segmentation is **skipped at inference** for latency reasons. The BiLSTM was trained on segmented text; unsegmented inference text could degrade quality. (Documented as a known limitation in the code comment.)

### D. ML/NLP Models

| Model | Architecture | Embeddings | Head | Parameters | Train Method |
|---|---|---|---|---|---|
| **Logistic Regression** | Linear | TF-IDF (uni+bigrams, max 40K vocab, sublinear_tf) | softmax | ~40K features | `liblinear/saga`, `C` grid-searched, `class_weight=balanced` |
| **SVM** | LinearSVC + CalibratedClassifierCV (default `use_linear=True`) | Same TF-IDF | calibrated probability | same features | `C` grid, 5-fold CV |
| **BiLSTM** | 1-layer bidirectional LSTM, hidden=128, dropout=0.3 | FastText `cc.vi.300.bin` (loaded if available) → fine-tuned | Linear(256→2) | ~7.5M (with FastText init) | AdamW, cosine LR, label smoothing, early-stopping (patience=5) |
| **PhoBERT** | `vinai/phobert-base` (RoBERTa, 12L/768H) + linear head | phobert tokenizer (BPE), max_len=256, dropout=0.1 | Linear(768→2) | ~135M | AdamW + **layer-wise LR decay (0.95)**, warmup 10%, batch=16 × grad_accum=4 (effective 64), epochs=8, patience=2, **layer_lr_decay is non-trivial engineering** |

**Optimizer / loss**: CrossEntropy with optional label smoothing, weighted CE with balanced class weights.

**Hyperparameter sweep**: only the LR/SVM C-parameter is grid-searched (5-fold). Deep models use one config each.

**Loss**: CrossEntropy (with optional label smoothing).

**Level assessment**:

- **Traditional ML (LR/SVM):** Solid — proper TF-IDF, calibration, grid search, balanced weighting.
- **BiLSTM:** **Intermediate** — sound implementation but with FastText initialization it's 2017-era technology.
- **PhoBERT:** **Intermediate-Advanced** — fine-tuning is correct (LR decay, warmup, grad accumulation). However, it uses a **single seed, single config** — no LoRA, no prompt tuning, no adapter, no hyperparameter sweep.

**Sophistication rating: Intermediate** (closer to Advanced for engineering quality, but only one seed/config per model, no LLM-era techniques).

### E. Evaluation

| Metric | Reported | Verified |
|---|---|---|
| Accuracy | ✅ | ✅ |
| Precision / Recall (macro & weighted) | ✅ | ✅ |
| F1 (macro & weighted, per-class) | ✅ | ✅ |
| ROC-AUC | ✅ (0.9495 for PhoBERT) | ✅ |
| Average Precision (PR-AUC) | ✅ | ✅ |
| Confusion matrix (per model + grid) | ✅ | ✅ |
| Per-class metrics | ✅ | ✅ |
| Classification report | ✅ | ✅ |
| Calibration (ECE/MCE/Brier) | ✅ | ✅ (in `calibration_analysis.py`) |
| Post-hoc calibration (Platt/Temp/Isotonic) | ✅ | ✅ (in `post_hoc_calibration.py`) |
| Reliability diagrams | ✅ | ✅ |
| **5-fold stratified CV × 3 seeds** | ✅ (LR/SVM only — deep models only single split) | ✅ (only traditional models) |
| **Bootstrap 95% CI (10,000 iter)** | ✅ | ✅ |
| **McNemar's test + Holm-Bonferroni correction** | ✅ | ✅ |
| **Cohen's d effect size** | ✅ | ✅ |
| **Token-level attribution (SHAP, IG, attention rollout)** | ✅ (Phase 1, all 4 models) | ✅ (`src/analysis/*`) |
| **Cross-model attribution agreement** | ✅ (faithfulness, rank agreement) | ✅ (`method_agreement.py`) |
| **Knowledge distillation F1-vs-size** | ✅ | ✅ |
| **Hard-cases deep dive** | ✅ | ✅ |
| **Ablation study (TF-IDF vocab, n-gram, segmentation, C, sublinear)** | ✅ (LR-only) | ✅ |

**What is missing for thesis-grade rigor:**

- ❌ **No temporal / concept-drift evaluation** (train on 2015 → test on 2017 split).
- ❌ **No cross-domain evaluation** (e.g. health vs politics vs sports).
- ❌ **No adversarial robustness study** (paraphrasing, synonym swap, etc.).
- ❌ **No statistical power analysis for ablation** (ablation only on LR; no significance tests on ablation deltas).
- ❌ **Single seed for deep models** — BiLSTM and PhoBERT results have no reported variance.
- ❌ **PR curve not reported per class**, only AP score.
- ❌ **No human evaluation / inter-annotator agreement** on the dataset labels themselves (the ground-truth quality is unverified).

**Verdict:** **Strong for the four-model comparison on this dataset.** Weak for generalisation claims.

### F. System Architecture

```
User → Streamlit (app.py)
 └─→ @st.cache_resource loads each model:
 ├─ LR (joblib pickle + TF-IDF vectorizer)
 ├─ SVM (joblib pickle + TF-IDF vectorizer)
 ├─ BiLSTM (torch state + vocab + optional FastText)
 └─ PhoBERT (torch state + HF tokenizer)
        └─→ preprocess_text() → predict_*() → render UI card

fakenews CLI (src/cli.py):
  preprocess → split → features → train → evaluate → calibration → explain (Phase1 SHAP/IG/attention)
              → recalibrate (Phase 2 Platt/Temp/Isotonic)
              → distill (Phase 3 KD)
              → hard-cases (Phase 4)
              → run (full pipeline)
```

No external services, no database, no API. Pure local batch + inference. Streamlit is the only UI.

### G. Engineering Quality

| Aspect | Verdict | Evidence |
|---|---|---|
| Code organization | **Excellent** | Clean `src/{preprocessing,features,models,training,evaluation,analysis}` layout; `pyproject.toml` with console-script `fakenews` |
| Modularity | **Excellent** | Each model has its own trainer class; shared `runner.py` eliminates boilerplate |
| Reproducibility | **Good** | `set_reproducibility_seeds()`, fixed seed=42; cudnn deterministic; `ExperimentTracker` logs config hash + git commit |
| Testing | **Good** | 17 test files (config, core, distillation, experiment_tracker, explainability_attribution, features, hard_cases, integration, metrics, models, recalibration, save_load, smoke, split_data, statistical_tests, text_preprocessor, utils) |
| Logging | **Good** | Centralised `get_logger` in `src/utils/logger.py`; per-epoch GPU telemetry via `gpu_monitor.py` |
| Error handling | **Adequate** | `try/except (ImportError, OSError, RuntimeError)` blocks around external dep (py_vncorenlp, FastText), `ArtifactMissingError` for the demo |
| Model versioning | **Light** | `ExperimentTracker` writes JSON; **no model registry, no DVC, no MLflow** |
| Configuration | **Excellent** | Centralised `config.py` dataclass; per-model hyperparameter dataclasses |
| Deployment | **Local-only** | `app.py` runs Streamlit; **no Dockerfile, no CI/CD, no cloud deployment** |
| Scalability | **N/A** | Single-user Streamlit, no batch serving |
| Dependency management | **Good** | `pyproject.toml` + `requirements.txt`; pinned versions |
| Linting | **Good** | ruff config in `pyproject.toml` |

---

## PHASE 2 — CURRENT PROJECT LEVEL

### Score (0–10)

| Dimension | Score | Justification |
|---|---|---|
| 1. Problem formulation | **6/10** | Binary fake-news defined but dataset provenance unknown; no explicit operationalisation of "fake" |
| 2. Dataset quality | **5/10** | 15K rows, balanced, but provenance unknown, 96.6% missing dates, ~5.6% exact dups, no IAA, mixed sources |
| 3. NLP methodology | **7/10** | PhoBERT + TF-IDF done correctly; word segmentation aligned with PhoBERT; Vietnamese handling is appropriate |
| 4. ML/model sophistication | **6.5/10** | Solid transformer fine-tuning with LR-decay + grad accum, but single seed; no LLM, no adapter/LoRA, no ensemble |
| 5. Experimental methodology | **7/10** | 5-fold CV × 3 seeds for traditional; McNemar + Holm-Bonferroni + bootstrap CI + Cohen's d; calibration analysis. But deep models = single split. |
| 6. Evaluation quality | **7/10** | Accuracy, F1, ROC-AUC, AP, confusion matrix, calibration (ECE/MCE/Brier), reliability diagrams, attribution. No temporal / domain / adversarial evaluation. |
| 7. Software engineering | **8/10** | Clean OOP, config centralisation, ExperimentTracker, decent tests. |
| 8. System architecture | **6/10** | CLI + Streamlit demo is fine. No API, no DB, no deploy infra. |
| 9. Reproducibility | **7/10** | Seeded, deterministic flags, ExperimentTracker. But raw dataset shared in repo (licence unclear), no environment lock-file beyond requirements. |
| 10. Academic contribution | **5.5/10** | Comparative study of 4 paradigms on a Vietnamese dataset is useful but not novel. KD experiment adds engineering novelty but not research novelty. No hypothesis-driven research question. |

**Average: 6.5/10**

### Current Project Level

> **Strong capstone project / Thesis-ready project (with one major caveat).**

**Why:**

- The engineering execution is well above a typical student project (config-driven, tests, logging, CLI, calibration, attribution, KD).
- The four-model comparison with statistical tests is publishable as a workshop paper.
- BUT the absence of a **single research hypothesis** + **unknown dataset provenance** keeps it from being *research-grade*.

**The gap to thesis-readiness** is the addition of a **research-grade experimental axis** that produces a **defensible, novel insight** — not just another comparison or another model.

---

## PHASE 3 — RESEARCH GAPS

### Gap 1 — Dataset provenance and label quality

- **Current limitation**: Source/licence unknown; ground-truth labels never audited.
- **Why it matters**: A thesis cannot defend results on an unverified dataset.
- **Current project does**: Trains 4 models on the data and reports numbers.
- **Missing**: Dataset card (Datasheets-for-Datasets style), source citation, inter-annotator agreement, label-noise estimation.
- **Research question**: *How does label noise (estimated via confident-learning) affect Vietnamese fake-news classifier rankings?*
- **Evaluation**: Compare 4 models on cleaned vs noisy labels; report rank stability.

### Gap 2 — No temporal / concept-drift evaluation

- **Current limitation**: Random 70/15/15 split ignores time. 96.6% of dates missing, but the 532 valid ones span 2015–2017 (notably old for a 2026 thesis).
- **Why it matters**: Real fake news evolves with new events, new entities, new writing styles.
- **Current project does**: Single random split.
- **Missing**: Time-based train/test splits.
- **Research question**: *How does the performance of Vietnamese fake-news classifiers degrade when evaluated on news published later than the training corpus?*
- **Evaluation**: Train on 2015–2016, test on 2017; report ΔF1, ΔROC-AUC.

### Gap 3 — Stylometric shortcut vs semantic understanding

- **Current limitation**: The dataset contains obvious surface cues (all-caps headlines, `<URL>` placeholders, hashtags, emoji). Models may learn these instead of credibility.
- **Why it matters**: A high-accuracy model that learned `<URL>` = fake is brittle.
- **Current project does**: No diagnostic for this.
- **Missing**: Feature ablation conditioned on style cues; per-cue performance breakdown.
- **Research question**: *Do current Vietnamese fake-news classifiers rely primarily on stylometric cues (URLs, hashtags, punctuation) rather than semantic content?*
- **Evaluation**: Replace/strip surface cues at test time; measure performance drop.

### Gap 4 — Single-seed deep-learning evaluation

- **Current limitation**: BiLSTM and PhoBERT results are point estimates with no confidence intervals.
- **Why it matters**: Claiming "PhoBERT > BiLSTM at p<0.05" requires multiple runs.
- **Current project does**: McNemar test on one run per model.
- **Missing**: 5+ seeds for each deep model, mean ± std, statistical test on seed distribution.
- **Research question**: *Are the reported rankings of Vietnamese fake-news classifiers stable across random seeds?*

### Gap 5 — No generalisation across domains/topics

- **Current limitation**: All articles are presumably mixed across topics; no per-topic performance breakdown.
- **Why it matters**: A model trained mostly on political fake news may fail on health fake news.
- **Current project does**: Aggregate metrics only.
- **Missing**: Topic-stratified evaluation.
- **Research question**: *How does classifier performance vary across Vietnamese news topics (health, politics, social, economics)?*

### Gap 6 — Absence of external evidence / fact-checking

- **Current limitation**: Model classifies text in isolation — no fact-checking, no retrieval augmentation, no knowledge-graph reasoning.
- **Why it matters**: This is the **next frontier** in fake-news research and is the strongest thesis-level contribution axis.
- **Current project does**: Pure supervised classification.
- **Missing**: Retrieval-Augmented Verification (Vietnamese trusted corpus → retrieval → NLI → aggregation).
- **Research question**: *Does retrieval-augmented evidence verification improve out-of-domain and adversarial-robustness performance of Vietnamese fake-news classifiers?*

### Gap 7 — No adversarial robustness study

- **Current limitation**: No test of how the model behaves under paraphrasing, synonym substitution, headline rewriting, AI-generated misinformation.
- **Why it matters**: Robustness is a real-world requirement.
- **Research question**: *How robust are Vietnamese fake-news classifiers to text-style perturbations that preserve meaning?*

### Gap 8 — No knowledge-graph / entity reasoning

- **Current limitation**: No NER, no entity-aware features, no KG-based reasoning.
- **Why it matters**: Structured signals (people, organisations, events) can complement text.
- **Research question**: *Do entity-relationship features improve Vietnamese fake-news detection over text-only transformers?*

### Gap 9 — No human-AI collaboration evaluation

- **Current limitation**: No measurement of whether explanations actually help humans.
- **Why it matters**: If the thesis claims XAI is useful, you must measure it.

---

## PHASE 4 — EIGHT THESIS DIRECTIONS (Vietnamese-only)

> All directions below are **scoped to Vietnamese only**. Multilingual / cross-lingual experiments are intentionally removed.

### Direction #1 — **Evidence-Based Vietnamese Fake News Detection with Retrieval + NLI**

**Research problem**: Current supervised classifiers learn surface cues; they fail when those cues are absent (paraphrased fake news) or absent at test time.

**Research question**: *Does retrieval-augmented evidence verification (claim extraction → Vietnamese trusted-corpus retrieval → NLI) improve out-of-domain and adversarial robustness of Vietnamese fake-news detection compared with text-only supervised classifiers?*

**Hypothesis**: A retrieval-augmented verifier achieves ≥2 F1 points on out-of-domain Vietnamese news and ≥5 F1 points on paraphrased-fake news than text-only PhoBERT, while remaining ≤1 F1 point worse on in-domain test.

**Technical approach**:

1. Claim extractor (PhoBERT-based) → identify verifiable claims.
2. Vietnamese search index over a trusted corpus (VnExpress, BBC Vietnamese, official government portals — **need to specify scope and document provenance**).
3. Retriever: BM25 + dense (Vietnamese multilingual encoder, e.g. BGE-M3 or `bkai-foundation-models/vietnamese-bi-encoder`) hybrid.
4. NLI model fine-tuned on Vietnamese NLI data (translated XNLI + synthetic Vietnamese pairs).
5. Aggregator: Logistic regression over per-evidence NLI scores + original-text classifier logits.
6. End-to-end evaluation on in-domain, out-of-domain, paraphrased Vietnamese test sets.

**Required changes**:

- New Vietnamese retrieval index module.
- New Vietnamese NLI module.
- New test sets (out-of-domain + adversarial).
- New aggregator.

**Expected academic contribution**: Establishes a **stronger** Vietnamese fake-news baseline than text-only classifiers; produces a reusable Vietnamese evidence corpus and benchmark.

**Experiments required**: In-domain, temporal-split, cross-topic, paraphrased-fake.

**Evaluation metrics**: F1, ROC-AUC, robustness gap Δ.

**Datasets required**: Existing + Vietnamese news corpus for retrieval + an out-of-domain test set + a paraphrased-fake set (created via back-translation or synonym swap).

**Difficulty**: 8/10 — *requires IR, NLI, Vietnamese index construction, careful latency management*.

**Implementation time**: High (6–10 weeks).

**Research value**: 9/10 — *publishable at a tier-1 venue (NAACL, EMNLP, ACM journals) if executed well; first-of-its-kind for Vietnamese*.

**Risk**: High — *depends on quality of Vietnamese retrieval corpus and Vietnamese NLI model availability*.

**How different from current project?**: Adds an entirely new pipeline (retrieval + NLI); current project is pure supervised classification.

**What would make this thesis-worthy?**: A defensible claim that *evidence-based verification generalises better than text-only classification on Vietnamese misinformation* — a novel finding for the Vietnamese-language community.

---

### Direction #2 — **Temporal Generalisation & Concept Drift of Vietnamese Fake News Classifiers**

**Research problem**: Fake news evolves with time, current models are evaluated on random splits.

**Research question**: *To what extent do state-of-the-art Vietnamese fake-news classifiers generalise across publication years, and what factors (vocabulary, entities, style) drive the performance drop?*

**Hypothesis**: Classifiers trained on Vietnamese news from year Y show ≥3-point F1 degradation on news from year Y+1, with the largest degradation attributed to new entities and changed writing style.

**Technical approach**:

1. Recover parseable dates (532 of 15,789); supplement with **public Vietnamese fake-news datasets that have dates** (e.g. ISE-FakeNews, FakeVN — to be verified at proposal time).
2. Time-based splits (train ≤ t₀, test > t₀).
3. Year-stratified evaluation; ΔF1 vs Δt.
4. Lexical / entity analysis of errors per time window.
5. Optional: simple drift detector (ADWIN, Page-Hinkley) on classifier confidence stream.

**Required changes**: Date-aware splitting; new analysis scripts; possibly new data.

**Expected academic contribution**: First systematic study of temporal generalisation in Vietnamese fake-news detection.

**Experiments required**: Time-based splits at 3+ cutoffs; confidence-drift plots.

**Evaluation metrics**: F1, ROC-AUC, ΔF1 vs time, calibration drift.

**Datasets required**: Existing + ISE-FakeNews / FakeVN-style temporal Vietnamese datasets (must verify availability).

**Difficulty**: 5/10.

**Implementation time**: Medium (4–6 weeks).

**Research value**: 8/10 — *addresses a known but under-studied issue; defensible thesis claim; clear narrative*.

**Risk**: Medium — depends on availability of dated Vietnamese fake-news data.

**How different?**: Adds a *temporal axis* absent from current project.

**Thesis-worthy?**: Yes, if the temporal split reveals a meaningful, explainable gap.

---

### Direction #3 — **Adversarial Robustness of Vietnamese Fake News Detection**

**Research problem**: Models are vulnerable to text perturbations; real-world adversaries paraphrase, swap synonyms, or use AI to rewrite fake news.

**Research question**: *How robust are PhoBERT-based Vietnamese fake-news classifiers to meaning-preserving text perturbations (paraphrasing, synonym substitution, character-level noise, AI-rewriting), and does adversarial training improve robustness without sacrificing in-domain accuracy?*

**Hypothesis**: Current models lose ≥5 F1 points under paraphrase attacks; adversarial training recovers ≥2 points while keeping in-domain F1 within 1 point.

**Technical approach**:

1. Build Vietnamese perturbation toolkit: back-translation (vi↔en↔vi), Vietnamese synonym swap (WordNet-vi or context-aware BERT-substitute using PhoBERT), character-level (homoglyph, swap, delete).
2. AI-rewriting subset: Vietnamese paraphraser (verify availability; possibly fine-tune a small Vietnamese T5).
3. Evaluate all 4 models on perturbed Vietnamese test sets.
4. Adversarial training: augment Vietnamese training set with perturbed examples.
5. Report F1, robustness gap, accuracy-efficiency tradeoff.

**Required changes**: Perturbation module; adversarial training pipeline.

**Expected academic contribution**: Quantifies the brittleness of Vietnamese fake-news classifiers and provides a robust-training recipe.

**Experiments required**: 5+ perturbation types × 4 models × perturbed-vs-clean F1.

**Evaluation metrics**: F1, ΔF1, character-error-rate, semantic-similarity (BERTScore) vs ΔF1.

**Datasets required**: Existing + perturbed versions.

**Difficulty**: 6/10.

**Implementation time**: Medium (4–6 weeks).

**Research value**: 7.5/10 — *a good topic; Vietnamese-language angle is the novelty*.

**Risk**: Medium.

**Thesis-worthy?**: Yes — Vietnamese-language adversarial robustness is under-studied.

---

### Direction #4 — **Stylometric Bias: Are Vietnamese Fake News Classifiers Learning Surface Cues?**

**Research problem**: Models may exploit hashtags, URLs, ALL-CAPS instead of content.

**Research question**: *To what extent do stylometric surface cues (URLs, hashtags, emoji, all-caps, punctuation density) drive Vietnamese fake-news classifiers' decisions, and does content-only evaluation reveal hidden weaknesses?*

**Hypothesis**: Removing Vietnamese surface cues drops F1 by ≥5 points, and SHAP attribution concentrates on the first 20% of tokens (most surface features).

**Technical approach**:

1. Build Vietnamese stylometric-feature extractors (URL count, hashtag count, emoji count, caps ratio, punctuation density, average word length).
2. Evaluate models on Vietnamese style-stripped test set (regex removal of URLs, hashtags, emoji).
3. SHAP / Integrated-Gradients analysis conditioned on style-feature presence (use existing `src/analysis/*` modules).
4. Train a "style-only" Vietnamese baseline (TF-IDF on style features only) and compare.
5. Mitigation: style-augmented training (counterfactual style stripping in training).

**Required changes**: Vietnamese stylometric extractor; style-stripped test set; analysis pipeline.

**Expected academic contribution**: First systematic quantification of stylometric bias in Vietnamese fake-news detection.

**Experiments required**: Style-stripped test × 4 models; SHAP analysis.

**Evaluation metrics**: F1, ΔF1, top-1 / top-5 attribution concentration, style-only baseline accuracy.

**Datasets required**: Existing.

**Difficulty**: 5/10.

**Implementation time**: Low–Medium (3–4 weeks).

**Research value**: 7/10 — *novel for Vietnamese; clear narrative; defensible experimental design*.

**Risk**: Low.

**Thesis-worthy?**: Yes — *clear, publishable insight*.

---

### Direction #5 — **Multimodal-Entity Knowledge-Graph Reasoning for Vietnamese Fake News**

**Research problem**: Pure text classifiers miss structured signals (entities, claims, source credibility).

**Research question**: *Does incorporating entity-relationship knowledge (persons, organisations, locations, claims) via a lightweight knowledge graph improve Vietnamese fake-news detection beyond text-only transformers?*

**Hypothesis**: Entity-aware features contribute ≥1 F1 point on the test set and provide more interpretable explanations.

**Technical approach**:

1. Vietnamese NER (`underthesea` or PhoBERT-NER) → entities (PER/ORG/LOC/MISC).
2. Relation extraction (simple rules or fine-tuned classifier).
3. Knowledge graph per article (entities + relations).
4. Hybrid encoder: text (PhoBERT) + graph (R-GCN or GraphSAGE).
5. Fusion layer; classification head.

**Required changes**: Vietnamese NER module, KG builder, graph encoder, hybrid model.

**Expected academic contribution**: A structured-reasoning model for Vietnamese fake news.

**Experiments required**: Text-only vs text+KG ablation.

**Evaluation metrics**: F1, ΔF1, attention-vs-graph-attribution analysis.

**Datasets required**: Existing + Vietnamese entity annotations.

**Difficulty**: 8/10.

**Implementation time**: High (6–8 weeks).

**Research value**: 7.5/10.

**Risk**: High — *Vietnamese NER quality on noisy text may be the bottleneck*.

**Thesis-worthy?**: Yes, if executed.

---

### Direction #6 — **Knowledge Distillation × Robustness (Engineering Contribution)**

**Research problem**: Already partially done in the project. But can be extended into a robustness study.

**Research question**: *Does knowledge distillation from PhoBERT to a compact Vietnamese BiLSTM preserve adversarial robustness, or does distillation amplify the teacher's brittle decisions?*

**Hypothesis**: Student model loses ≥2 F1 points of robustness vs teacher; adversarial fine-tuning of the student recovers ≥1 point.

**Technical approach**:

1. Use the existing Vietnamese KD pipeline as baseline.
2. Generate perturbed teacher logits (perturbed-input → teacher-logits → student training targets).
3. Compare student distilled from clean teacher vs perturbed teacher.
4. Evaluate both students on clean + perturbed Vietnamese test sets.

**Required changes**: Perturbation-aware KD pipeline.

**Expected academic contribution**: Insight into whether distillation preserves robustness — a question of broader NLP interest.

**Experiments required**: Clean vs perturbed teacher distillation × clean/perturbed Vietnamese test.

**Evaluation metrics**: F1, robustness gap, calibration (ECE).

**Datasets required**: Existing + perturbed Vietnamese test set.

**Difficulty**: 4/10.

**Implementation time**: Low (2–3 weeks).

**Research value**: 6.5/10 — *extends existing work but limited novelty*.

**Risk**: Low.

**Thesis-worthy?**: As a *secondary* contribution, not as a standalone thesis.

---

### Direction #7 — **Vietnamese LLM Zero/Few-Shot vs Fine-Tuned Detection**

**Research problem**: Are Vietnamese LLMs (PhoGPT, ViGPT, VinaLLaMA, Qwen2.5-VL) competitive with fine-tuned PhoBERT without any task-specific training?

**Research question**: *How do general-purpose Vietnamese (and Vietnamese-fine-tuned) LLMs, used in zero-shot and few-shot settings, compare with fine-tuned PhoBERT on Vietnamese fake-news detection across in-domain, cross-domain, and adversarial test sets?*

**Hypothesis**: Zero-shot Vietnamese LLMs are within 5 F1 points of fine-tuned PhoBERT in-domain but degrade more on adversarial test; few-shot (k=10) closes the gap to within 2 F1 points.

**Technical approach**:

1. Pick 2-3 Vietnamese LLMs: PhoGPT, VinaLLaMA (or Qwen2.5-7B-Instruct as multilingual baseline).
2. Zero-shot prompting with calibrated Vietnamese instruction template.
3. Few-shot k=5, k=10 with representative Vietnamese examples.
4. Compare with PhoBERT on in-domain, cross-domain (topic), adversarial Vietnamese test sets.

**Required changes**: LLM serving module (vLLM / HF pipeline), Vietnamese prompt engineering, evaluation harness.

**Expected academic contribution**: First systematic Vietnamese LLM-vs-fine-tuned comparison for fake-news detection.

**Experiments required**: 3 LLMs × {zero, k=5, k=10} × {in-domain, cross-topic, adversarial}.

**Evaluation metrics**: F1, ROC-AUC, latency, cost.

**Datasets required**: Existing + perturbed Vietnamese test sets.

**Difficulty**: 5/10.

**Implementation time**: Medium (3–5 weeks).

**Research value**: 7/10 — *timely; LLM-era relevance*.

**Risk**: Medium — *LLM availability and reproducibility of results*.

**Thesis-worthy?**: Yes, but feels more like a benchmarking paper than a research thesis unless paired with strong analysis.

---

### Direction #8 — **Cross-Domain / Cross-Topic Generalisation (Vietnamese)**

**Research problem**: Real Vietnamese fake news spans many topics; training on one topic may fail on another.

**Research question**: *To what extent does Vietnamese fake-news detection transfer across topics (politics, health, social, economics, entertainment), and which model generalises best?*

**Hypothesis**: Topic-stratified evaluation shows ≥4-point F1 variance across Vietnamese topics; PhoBERT with topic-adversarial training reduces the variance by ≥1.5 points.

**Technical approach**:

1. Manually annotate or auto-label the Vietnamese test set by topic (rule-based classifier or Vietnamese LLM).
2. Topic-stratified evaluation of all 4 models.
3. Topic-adversarial training (gradient reversal layer).
4. Cross-topic confusion matrix.

**Required changes**: Topic labelling; topic-adversarial model.

**Expected academic contribution**: First cross-topic study for Vietnamese fake news.

**Experiments required**: Leave-one-topic-out CV; topic confusion.

**Evaluation metrics**: F1, ROC-AUC, transfer matrix.

**Datasets required**: Existing + Vietnamese topic labels.

**Difficulty**: 6/10.

**Implementation time**: Medium (4–6 weeks).

**Research value**: 7/10.

**Risk**: Medium — depends on Vietnamese topic-labelling quality.

**Thesis-worthy?**: Yes, combined with topic-adversarial training as a contribution.

---

## PHASE 5 — RESEARCH AXES SUMMARY

| Axis | Status in current project | Best direction |
|---|---|---|
| Model architecture (more Vietnamese transformers) | PhoBERT done; ViT5, BARTpho not tested | **#7 (Vietnamese LLM comparison)** |
| Explainable AI | SHAP/IG/attention implemented but not user-evaluated | **Extend Direction #4** |
| Multimodal / KG | Text only | **#5 (KG)** |
| Retrieval-augmented / evidence | None | **#1 (RAG + NLI)** ⭐ |
| Knowledge Graph | None | **#5** |
| Temporal / concept drift | None | **#2** ⭐ |
| Cross-domain | None | **#8** |
| Adversarial robustness | None | **#3** |
| LLM-based detection | None | **#7** |
| Human-AI collaboration | None | (Out of scope for Vietnamese-only unless a small user study is feasible) |

---

## PHASE 6 — ACADEMIC CONTRIBUTION ANALYSIS

| Direction | Engineering contribution | Research contribution |
|---|---|---|
| **#1 Evidence-Based** | Vietnamese RAG + NLI pipeline | Establishes a stronger, generalisable baseline for Vietnamese fake-news |
| **#2 Temporal** | Vietnamese time-based split framework | Quantifies concept drift in Vietnamese fake news |
| **#3 Adversarial** | Vietnamese perturbation toolkit, robust training | Quantifies brittleness + offers mitigation |
| **#4 Stylometric bias** | Vietnamese style-stripped evaluation harness | Reveals hidden shortcut learning |
| **#5 KG** | Hybrid text+graph Vietnamese model | Structured-reasoning baseline |
| **#7 LLM comparison** | Vietnamese LLM evaluation harness | Zero/few-shot ceiling for Vietnamese fake-news |
| **#8 Cross-domain** | Vietnamese topic labels, adversarial training | First cross-topic study |

**Strongest research contribution candidates**: #1, #2, #4. Each is publishable in its own right and 100% Vietnamese.

---

## PHASE 7 — EXPERIMENTAL DESIGN (TOP 3 DIRECTIONS)

I'll design for **Direction #1 (RAG-NLI)**, **Direction #2 (Temporal)**, **Direction #4 (Stylometric Bias)**.

### A. Direction #1: Evidence-Based Vietnamese Fake News Detection (RAG + NLI)

#### Baseline

- **PhoBERT fine-tuned on the existing Vietnamese training set** (already in repo). **Same hyper-parameters**, **same random seed**.

#### Improved Model

- **Pipeline A — BM25 retrieval + Vietnamese NLI**:
  1. Claim extraction: take the first 2 sentences of the article as the claim.
  2. BM25 retrieval over a Vietnamese trusted corpus (e.g. VnExpress archive, official government RSS feeds — **must verify and document provenance**).
  3. Top-5 evidence passages.
  4. NLI: fine-tune `xlm-roberta-base` or a Vietnamese encoder on translated XNLI + synthetic Vietnamese pairs.
  5. Aggregator: logistic regression over {claim–evidence NLI scores, max-refutes, mean-refutes, no-evidence flag, original PhoBERT logit}.

- **Pipeline B — Dense retrieval (BGE-M3 or `bkai-foundation-models/vietnamese-bi-encoder`) + ViT5-NLI**: same structure with dense retriever.

#### Ablation Study

| Ablation | Question |
|---|---|
| A1: No claim extraction (use whole article as query) | Does claim decomposition help? |
| A2: BM25 only (no dense) | Does dense retrieval matter? |
| A3: Top-1 evidence only | Is top-k >1 necessary? |
| A4: No aggregator (use max-NLI score) | Does learned aggregation matter? |
| A5: No original-text logit | Does the text classifier add value over evidence? |
| A6: Replace NLI with simple lexical overlap | Is NLI necessary? |
| A7: Train NLI on synthetic only vs include XNLI | Does multilingual NLI transfer to Vietnamese? |

#### Dataset Split

- **In-domain train/val/test**: same split as the existing project (random stratified 70/15/15).
- **Out-of-domain test**: collect ~500 Vietnamese articles from a different time period or topic (politics-only for test if training was mixed).
- **Paraphrased test**: back-translate each test article vi → en → vi and re-test.
- **Adversarial test**: synonym-swap variant of 200 test articles.

#### Metrics

- F1-macro, ROC-AUC, average precision.
- **Robustness gap**: ΔF1 between in-domain and adversarial.
- **Evidence recall**: of articles where retrieval found ≥1 relevant passage.
- **NLI accuracy**: on a held-out manually-labelled Vietnamese claim–evidence subset (200 pairs).
- Calibration: ECE, Brier.

#### Statistical Evaluation

- 5 random seeds for the aggregator (LR over NLI scores) and for the NLI fine-tuning.
- Bootstrap 95% CI on F1.
- Paired t-test or Wilcoxon on F1 across seeds for PhoBERT vs RAG-PhoBERT.

#### Error Analysis

- Per-class confusion matrix.
- Stratify errors by retrieval recall (with/without evidence).
- Sample 30 false positives and 30 false negatives; manually diagnose.

#### Robustness

- Paraphrased-fake test (back-translation).
- Synonym-swap test (BERT-substitute).
- Character-level noise test.
- AI-rewriting test (if a Vietnamese paraphraser is available).

---

### B. Direction #2: Temporal Generalisation (Vietnamese)

#### Baseline

- PhoBERT trained on **random** split (existing).

#### Improved Model

- PhoBERT trained on **time-ordered** split with explicit re-tuning (e.g. weight decay schedule).
- Optional: temporal fine-tuning with rolling re-init.

#### Ablation

- B1: Random split, time-test (lower bound).
- B2: Time-split train, time-test.
- B3: Time-split train + recent-data up-weighting.
- B4: Time-split train + chronological validation set (vs random validation).
- B5: B4 + EMA of model weights across time.

#### Dataset Split

- Reconstruct dates (only 532 are parseable; document this gap honestly in the thesis).
- Cutoffs at three points (2015-Q4, 2016-Q2, 2016-Q4).
- For each cutoff: train on ≤ cutoff, test on > cutoff.

#### Metrics

- F1, ΔF1 vs Δt, ROC-AUC.
- **Calibration drift**: ECE computed per quarter.
- **Entity novelty**: % of test entities unseen in training (proxy for "new event").

#### Statistical Evaluation

- Multi-seed (5 seeds) for each time split.

#### Error Analysis

- Lexical change: vocabulary overlap between train and test (Jaccard).
- Entity change: % new entities per test sample.

---

### C. Direction #4: Stylometric Bias (Vietnamese)

#### Baseline

- PhoBERT (full).

#### Improved Model

- Style-augmented training: during training, randomly strip URL/hashtag/emoji/caps from a fraction of Vietnamese inputs.

#### Ablation

- C1: Style-stripped test set, model evaluated on it.
- C2: Style-only baseline (TF-IDF on style features only).
- C3: Style-stripped training + style-stripped test.
- C4: Mixed style-preserved + style-stripped training (50/50).

#### Dataset Split

- Same as existing.

#### Metrics

- F1 on full vs style-stripped test.
- ΔF1 per cue type (URL, hashtag, emoji, caps, punctuation).
- Top-k token concentration in SHAP attribution (are they surface tokens?).

#### Statistical Evaluation

- 5 seeds.

#### Error Analysis

- Feature ablation: test on synthetic Vietnamese examples where content is true but style = fake, and vice versa.

---

## PHASE 8 — RANKING

Scoring 1–10 per criterion (higher = better).

| Rank | Direction | Academic Value | Technical Difficulty | Novelty | Feasibility | Thesis Potential | Risk |
|---|---|---|---|---|---|---|---|
| **🥇 1** | **#1 RAG + NLI** | **10** | 8 (hard) | **10** | 5 | **10** | High |
| **🥈 2** | **#2 Temporal Generalisation** | **9** | 5 | 9 | **8** | **9** | Medium |
| **🥉 3** | **#4 Stylometric Bias** | 8 | 4 | **9** | **9** | 8 | Low |
| 4 | #3 Adversarial Robustness | 8 | 6 | 7 | 7 | 7.5 | Medium |
| 5 | #5 KG Reasoning | 7.5 | 8 | 8 | 5 | 7 | High |
| 6 | #8 Cross-domain | 7 | 6 | 7 | 6 | 7 | Medium |
| 7 | #7 Vietnamese LLM Comparison | 7 | 5 | 7 | 7 | 6.5 | Medium |
| 8 | #10 Stylometry + Transformer | 6 | 5 | 6 | 7 | 6 | Low |
| 9 | #6 KD Robustness | 6.5 | 4 | 6 | 9 | 5.5 | Low |

### 🥇 Best Overall: **Direction #1 — Evidence-Based Vietnamese Fake News Detection (RAG + NLI)**

**Why**: Highest academic value, highest novelty, most defensible thesis narrative. The retrieval-augmented verifier is a *qualitative shift* from text-only classification and aligns with the international state-of-the-art (FEVER, ClaimDecomp, WiCE). For Vietnamese specifically, it would be a true first-of-its-kind contribution.

### 🥈 Strong Alternative: **Direction #2 — Temporal Generalisation**

**Why**: Highest feasibility, still strong research value. Avoids the need to build an entirely new retrieval infrastructure. Honest, well-scoped, and directly answers the "does the model actually work in deployment?" question.

### 🥉 Safe Alternative: **Direction #4 — Stylometric Bias**

**Why**: Lowest risk, lowest difficulty, still novel for Vietnamese. Can be completed within a single semester and produces a clear, publishable insight. Best choice if time-constrained.

---

## PHASE 9 — PROPOSED FINAL THESIS

For Direction #1 (RAG + NLI), the strongest direction:

### Thesis Title (3 candidates)

1. **"Evidence-Based Vietnamese Fake News Detection: A Retrieval-Augmented Approach Using Claim Decomposition and Natural Language Inference"**
2. **"Beyond Text Classification: Retrieval-Augmented Verification for Robust Vietnamese Fake News Detection"**
3. **"From Surface Cues to Semantic Evidence: Improving Robustness of Vietnamese Fake News Detection through Retrieval and NLI"**

### Problem Statement

Supervised Vietnamese fake-news classifiers achieve high accuracy on in-domain test sets (>90% F1 with PhoBERT), but their robustness under paraphrase attacks, out-of-domain topics, and adversarial perturbations is unknown. More importantly, their decisions may be driven by surface cues (URLs, hashtags, ALL-CAPS) rather than semantic veracity. This thesis investigates whether **retrieval-augmented verification** — explicitly grounding each prediction in trusted Vietnamese evidence — yields more robust and more interpretable fake-news detection than text-only supervised classification.

### Research Questions

1. **RQ1**: Does retrieval-augmented evidence verification improve out-of-domain performance of Vietnamese fake-news detection compared with text-only PhoBERT?
2. **RQ2**: Does retrieval-augmented verification improve adversarial robustness (paraphrase, synonym swap, AI-rewriting) of Vietnamese fake-news detection?
3. **RQ3**: How does the choice of retriever (BM25 vs dense Vietnamese encoder) and NLI backbone (XLM-R vs ViT5 vs PhoBERT-based Vietnamese NLI) affect evidence-based Vietnamese fake-news detection?
4. **RQ4**: Does evidence-based detection produce more interpretable, human-verifiable explanations than attribution-only methods (SHAP, IG)?
5. **RQ5**: What are the failure modes of retrieval-augmented detection on Vietnamese news (e.g. when retrieval fails or evidence is unavailable)?

### Hypotheses

- **H1**: A retrieval-augmented verifier achieves ≥2 F1 points higher than text-only PhoBERT on out-of-domain Vietnamese news.
- **H2**: Retrieval-augmented verification loses ≤2 F1 points under paraphrase attacks, vs ≥5 points for text-only PhoBERT.
- **H3**: A dense Vietnamese retriever outperforms BM25 by ≥1.5 F1 points on Vietnamese evidence retrieval.
- **H4**: Aggregating claim-level NLI scores with the original classifier's logit improves over either signal alone.
- **H5**: Evidence-grounded explanations yield ≥10% higher user-rated helpfulness than SHAP attribution in a small-scale user study (n≥30).

### Objectives

**General objective**: To investigate whether retrieval-augmented evidence verification yields more robust and interpretable Vietnamese fake-news detection than text-only supervised classification.

**Specific objectives**:

1. Construct a Vietnamese evidence corpus from trusted sources (≥100K articles).
2. Implement and evaluate a BM25 and a dense Vietnamese retriever on Vietnamese evidence retrieval.
3. Fine-tune a Vietnamese NLI model on multilingual + synthetic Vietnamese data.
4. Build a retrieval-augmented verifier and evaluate against text-only baselines on in-domain, out-of-domain, and adversarial Vietnamese test sets.
5. Conduct a small user study to evaluate the interpretability of evidence-grounded explanations.

### Proposed Architecture

```
[Input Vietnamese article]
    ↓
[Claim Extractor: PhoBERT-fine-tuned for span detection]
    ↓
[Claims list]
    ↓
[Retriever: BM25 + Vietnamese dense encoder hybrid] ← [Vietnamese Trusted Corpus Index]
    ↓
[Top-k Evidence Passages]
    ↓
[NLI: Fine-tuned XLM-R / ViT5 / Vietnamese encoder] ← predict ENTAILS / REFUTES / NEI per (claim, evidence)
    ↓
[Aggregator: LR / MLP over (NLI scores, original PhoBERT logit)]
    ↓
[Final Label: Real / Fake + confidence + evidence chain]
```

### Dataset

- **Existing** 15,789 Vietnamese news articles (with proper provenance documentation).
- **New**: ≥100K-article Vietnamese trusted corpus for retrieval.
- **New**: ≥500-article out-of-domain Vietnamese test set.
- **New**: ≥300-article paraphrased-fake Vietnamese test set (back-translated + AI-rewritten).

### Experimental Setup

1. **In-domain baseline**: PhoBERT (existing).
2. **Retrieval-augmented**: 2 retriever variants × 2 NLI variants × 2 aggregator variants = 8 configurations.
3. **Out-of-domain test**: Vietnamese articles from a different time period or topic.
4. **Adversarial test**: Vietnamese paraphrase, synonym-swap, AI-rewriting.
5. **User study** (optional): 30 participants × 50 Vietnamese articles (with/without evidence).

### Evaluation

- **Primary**: F1-macro, ROC-AUC.
- **Robustness**: ΔF1 across perturbation types.
- **Retrieval**: Recall@k, MRR.
- **NLI**: accuracy on held-out manual Vietnamese annotations.
- **Calibration**: ECE, Brier.
- **User study** (optional): F1, decision time, helpfulness rating.

### Expected Contributions

**Technical**:

- Vietnamese evidence corpus (reusable benchmark).
- Vietnamese NLI fine-tuning recipe.
- Retrieval-augmented Vietnamese fake-news detection pipeline.

**Research**:

- First systematic study of evidence-based detection for Vietnamese fake news.
- Quantification of robustness gains over text-only baselines.
- Identification of failure modes of retrieval-augmented verification.

**Practical**:

- A deployable API for fact-checking Vietnamese articles with cited evidence.
- Reusable Vietnamese benchmarks and corpora for the Vietnamese NLP community.

---

## PHASE 10 — IMPLEMENTATION ROADMAP (for Direction #1)

### Phase 0 — Understand Current System (1 week)

**Goal**: Document the existing pipeline.
**Tasks**: Read existing code; document Vietnamese data provenance; reproduce baseline metrics.
**Deliverables**: `docs/current_system.md`; reproduced baseline F1 numbers.
**Dependencies**: none.
**Estimated effort**: 1 week.
**What can go wrong**: dataset licence unknown.

### Phase 1 — Establish Baseline (1 week)

**Goal**: Lock in reproducible PhoBERT baseline.
**Tasks**: 5-seed runs of LR/SVM/BiLSTM/PhoBERT; report F1 ± std.
**Deliverables**: `results/baseline_5seed.csv`.
**Dependencies**: Phase 0.
**Estimated effort**: 1 week.

### Phase 2 — Dataset Improvements (2–3 weeks)

**Goal**: Document provenance; build Vietnamese retrieval corpus.
**Tasks**:

- Write a Datasheet for the existing Vietnamese dataset.
- Collect ≥100K Vietnamese news articles from VnExpress, BBC Vietnamese, official government RSS.
- Index corpus with BM25 + Vietnamese dense encoder.

**Deliverables**: `data/datasheet.md`; `data/trusted_corpus/`; BM25 index; dense index.
**Dependencies**: scraping agreements (must verify terms of service).
**Estimated effort**: 2–3 weeks.

### Phase 3 — Research Model: Retrieval + NLI (4 weeks)

**Goal**: Build retrieval-augmented Vietnamese verifier.
**Tasks**:

- Claim extractor (fine-tune PhoBERT on synthetic Vietnamese claim-extraction data or use heuristic first-2-sentences).
- BM25 retriever integration.
- Dense retriever integration (Vietnamese dense encoder).
- Fine-tune Vietnamese NLI model (XLM-R-base on XNLI + synthetic Vietnamese pairs).
- Aggregator: LR over (NLI scores, original PhoBERT logit).

**Deliverables**: `src/pipeline/rag_verifier.py`; trained Vietnamese NLI model.
**Dependencies**: Phase 2.
**Estimated effort**: 4 weeks.
**What can go wrong**: Vietnamese NLI fine-tuning may underperform; dense retrieval may not generalise; Vietnamese NLI datasets are scarce.

### Phase 4 — Experiments (3 weeks)

**Goal**: Run all experimental conditions.
**Tasks**:

- In-domain test (2 model variants × 5 seeds × 8 configurations = 80 runs).
- Out-of-domain Vietnamese test.
- Vietnamese adversarial test (paraphrase, synonym swap).
- User study (small-scale, optional).

**Deliverables**: `results/experiments/*.csv`; `paper/figures/*`.
**Dependencies**: Phase 3.

### Phase 5 — Analysis (2 weeks)

**Goal**: Interpret results.
**Tasks**:

- Attribution analysis (which Vietnamese evidence contributed).
- Calibration analysis (ECE).
- Failure-mode analysis.
- Statistical tests (paired bootstrap, McNemar).

**Deliverables**: `paper/analysis.md`; figures.

### Phase 6 — Backend Integration (1 week)

**Goal**: Wrap in API.
**Tasks**: FastAPI service; load trained Vietnamese models; expose /predict endpoint.
**Deliverables**: `api/main.py`.

### Phase 7 — Frontend / Demo (1 week)

**Goal**: User-facing Vietnamese demo.
**Tasks**: Extend Streamlit app to call RAG-verifier; show Vietnamese evidence chain; show explanation.
**Deliverables**: updated `app.py`.

### Phase 8 — Evaluation Consolidation (1 week)

**Goal**: Finalise numbers for thesis.
**Tasks**: Reproduce all numbers; finalise tables; freeze code at git tag.

### Phase 9 — Thesis Writing (4–6 weeks)

**Goal**: Write thesis document.
**Tasks**: Introduction, Related Work, Method, Experiments, Discussion, Conclusion.
**Deliverables**: `thesis/main.pdf`.

---

## 🔥 Final Advisor Recommendation

Here is my brutally honest assessment.

### 1. What to KEEP from the current project

- **The 4-model comparative framework** (LR, SVM, BiLSTM, PhoBERT) is a solid *engineering baseline* that demonstrates your modelling competence. Use it as **Chapter 4 (Baseline study)** of your thesis.
- **The Vietnamese preprocessing pipeline** (py_vncorenlp + cleaning + dedup) — it's correct and well-documented.
- **The statistical-testing suite** (McNemar + Holm-Bonferroni + bootstrap + Cohen's d) — keep it; extend it.
- **The ExperimentTracker** — useful for the thesis writeup.
- **The knowledge-distillation module** (Phase 3) — repurpose it for your ablation if you adopt Direction #1.
- **The explainability modules** (`lr_svm_shap`, `bilstm_attribution`, `phobert_attribution`, `method_agreement`) — extremely valuable; lift them into the new pipeline.
- **The Streamlit demo** — extend it to surface Vietnamese evidence chains and become the demonstration chapter.

### 2. What to REMOVE or stop spending time on

- **Do not** spend more time on **yet another transformer variant** (RoBERTa-large, DeBERTa, etc.). The marginal accuracy gain is not a thesis contribution.
- **Do not** polish the calibration plot or the post-hoc calibration further. ECE numbers are sufficient for a thesis.
- **De-prioritise the F1-vs-size KD plot** as a primary contribution. Use it as a *secondary* engineering result.
- **Stop expanding the dataset-preprocessing rules**. The current pipeline is fine.
- **Do not** pursue multilingual / cross-lingual extensions. You scoped this thesis to Vietnamese only.

### 3. What to IMPROVE

- **Dataset provenance documentation**. This is a glaring gap. You must write a Datasheets-for-Datasets document. Even if the source is "UNKNOWN", document what you know (collection date range, observed characteristics, label distribution, length statistics, presence of URLs/hashtags/emoji) — this is exactly what a thesis examiner will ask.
- **Reproducibility**: lock a final commit; provide `requirements.txt` hashes; document hardware used.
- **Multi-seed deep-model evaluation** (currently BiLSTM and PhoBERT have one seed). Even 3 seeds will materially strengthen your claims.
- **Calibration of deep models** (currently ECE reported for all 4; ensure it is in the thesis).
- **Write an "Operationalisation of Fake News" paragraph** — explicitly define what you mean by "fake" (verifiable misinformation, satire, opinion, etc.) and acknowledge the dataset's possible conflations.

### 4. What new RESEARCH COMPONENT to add

**My strong recommendation**: Direction #1 (Evidence-Based RAG + NLI) with a Vietnamese-only trusted corpus, with a serious temporal-generalisation study (Direction #2) as a *secondary* research axis.

**Concretely**:

- Build a **Vietnamese trusted corpus** (≥100K articles) from openly accessible Vietnamese news sources (VnExpress, BBC Vietnamese, Tuoi Tre, official government RSS). Document the selection.
- Implement **BM25 + dense retrieval** using a Vietnamese dense encoder.
- Fine-tune a **Vietnamese NLI model** on multilingual XNLI + synthetic Vietnamese pairs.
- Build a **retrieval-augmented verifier** that combines (claim → evidence → NLI → aggregator).
- Evaluate on **in-domain**, **out-of-domain**, and **paraphrased** Vietnamese test sets.
- **Report all baselines with multi-seed uncertainty.**

If you cannot commit to a full retrieval corpus, **Direction #2 (Temporal Generalisation)** is the strongest fallback. It addresses a real, under-studied question in Vietnamese fake-news research and is highly feasible.

### 5. What experiments you absolutely need

Regardless of which direction you choose, your thesis must contain:

1. **5-seed deep-model evaluation** (BiLSTM and PhoBERT) with mean ± std.
2. **Multi-comparison-corrected statistical tests** (already done — keep).
3. **Out-of-distribution test** — either temporal (Direction #2), topic-based (Direction #8), or paraphrase-based (Direction #3). One of these is non-negotiable.
4. **Robustness test** (paraphrase, synonym swap) — even a minimal version.
5. **Calibration evaluation** (ECE, reliability diagrams) — already done.
6. **Ablation study with statistical comparison** — extend the existing LR-only ablation to the deep models.
7. **Error analysis** with qualitative diagnosis of Vietnamese failure modes (already partly done — extend it).

### 6. What would make this look like a real graduation thesis

- **One clear research question** (not "compare 4 models") — e.g. "Does evidence-based retrieval improve Vietnamese fake-news robustness?"
- **A stated hypothesis with operational metrics** — e.g. "≥2 F1 points on out-of-domain test, p<0.05".
- **A new artefact the thesis produces** (Vietnamese corpus, Vietnamese pipeline, Vietnamese benchmark) that other Vietnamese NLP researchers can use.
- **Multi-seed uncertainty** reported honestly.
- **A defensible Vietnamese dataset provenance** document.
- **A discussion of limitations** that goes beyond "more data would help" — discuss **concept drift**, **style shortcut learning**, **label noise**, **evaluation gaming**.
- **A reproducible code release** with seed-locked experiments.
- **An actual experimental contribution** that is **novel for Vietnamese**, not just "we did X with PhoBERT".

### 7. What direction I would personally choose and WHY

**I would choose Direction #2 (Temporal Generalisation)** as the primary contribution and **Direction #4 (Stylometric Bias)** as a secondary contribution.

**Why Direction #2 over Direction #1**:

- **Feasibility**: A full retrieval-augmented Vietnamese system requires corpus construction, NLI fine-tuning, and dense-retriever infrastructure — easily a 6-month undertaking for a student.
- **Risk**: Retrieval quality and NLI quality for Vietnamese are not guaranteed. A weak retrieval pipeline produces *worse* results than a strong text classifier, and the thesis risk becomes "did retrieval actually help or hurt?"
- **Defensibility**: Temporal generalisation is straightforward to explain, visualise, and defend. The thesis narrative — "we trained on Vietnamese news from year X, but does it work on year Y?" — is intuitive for any examiner.
- **Novelty**: For Vietnamese specifically, temporal generalisation of fake-news classifiers has **not been studied**. This is a clear research gap.

**Why also Direction #4 (Stylometric Bias)**:

- It can be completed in 2–3 weeks.
- It produces a publishable insight *and* an immediate action ("we should strip style cues before training").
- It pairs naturally with Direction #2: stylometric shortcuts are amplified over time as fake-news producers adapt style.

**If you have time and energy**, you can layer in a **small retrieval-based component** (Direction #1) as a single chapter, but it should not be the thesis's spine.

---

## Final Verdict

> Your current project is a **strong capstone / borderline thesis-ready engineering project** with a Vietnamese focus. The engineering execution is genuinely impressive for a student. But as a thesis, it lacks a single research hypothesis, a clear contribution, and dataset provenance. To graduation-thesis level, you must **add one new research axis** (Direction #1, #2, or #4) and **document Vietnamese dataset provenance honestly**.
>
> If you only have time for one thing: add **multi-seed evaluation of BiLSTM and PhoBERT + a Vietnamese temporal (or topic) out-of-distribution test**. That single addition transforms this from "good engineering project" to "thesis-ready work".
>
> If you have a full semester: pursue **Direction #2 (Temporal Generalisation)** as the spine, with **Direction #4 (Stylometric Bias)** as a secondary chapter. That gives you a defensible, novel, feasible, and well-scoped graduation thesis — fully scoped to Vietnamese.
>
> If you have two semesters and want a publishable artefact: pursue **Direction #1 (Vietnamese RAG + NLI)**. This is the highest-impact direction but requires significant Vietnamese-specific infrastructure (trusted corpus + Vietnamese NLI model + retriever infrastructure).

---

*"Not verified from the current codebase" markers used where appropriate. The audit is complete."*