# 🧪 RESEARCH ARCHITECT REPORT — Vietnamese Fake News Detector

**Repository inspected**: `FakeNewsDetector` (commit on disk, 2026-09-15).
**Scope**: Vietnamese-only.
**Audience**: thesis advisor + the student.
**Output role**: senior ML researcher + research architect + thesis advisor.

---

## 0. Critical Findings from Re-Inspection

Before any architecture, here is what I verified **directly from the codebase**. These are facts — not assumptions.

### F1. Dataset schema is *minimal*

```
columns = ['id', 'text', 'date', 'label']
n_rows  = 15,789 raw → 13,958 after cleaning (886 exact duplicates removed)
labels  = 8,355 Real (52.9%) | 7,434 Fake (47.1%)
dates   = only 532 parseable (96.6% missing); range 2015-04 → 2017-12
```

There is **NO** source field, NO author field, NO URL field, NO category field. Any architectural direction that assumes these fields will fail.

### F2. The dataset is structurally heterogeneous

Inspection of real samples vs. fake samples reveals a striking pattern:

| Aspect | Real (label 0) | Fake (label 1) |
|---|---|---|
| Writing style | Formal news (Tuoi Tre, VnExpress, official channels) | Twitter/Facebook posts, opinion, satire |
| Mean word count | 85.7 | 115.5 |
| Mean sentences | 4.3 | 5.9 |
| Mean exclamation marks | 0.09 | **0.27** (3× more) |
| Attribution markers ("Theo", "Bộ Y tế", "TTXVN", "Reuters") | 666/8355 (8%) | 313/7434 (4%) |
| Articles with URL/mention | (mostly) absent | higher |

**Implication**: the dataset is *partially* a **style classification problem disguised as a fake-news problem**. The label is correlated with writing register. This is **an empirical fact** the model can exploit, and a thesis must address it.

### F3. URL is a near-perfect label leak (raw, before cleaning)

- 85.8% of articles with URL are Fake.
- 14.2% of articles without URL are Fake.
- 82.6% of articles with `@mention` are Fake.
- 81.3% of articles with `#hashtag` are Real (opposite direction).

The current preprocessing already strips URLs (`re.sub(r'https?://\S+|www\.\S+', '', text)`), hashtags remain via `\w` match, and `@mentions` are kept as-is (they look like words). This means the leakage is partially mitigated but **not eliminated** — word patterns around URLs/mentions still exist.

### F4. Existing models

- **PhoBERT** (`vinai/phobert-base`): standard `AutoModelForSequenceClassification`, fine-tuned with layer-wise LR decay (0.95), AdamW, 8 epochs, patience 2, batch 16 × grad accum 4 (effective 64), label smoothing 0.0, max_seq_len 256. **Single seed (42).**
- **BiLSTM**: 1-layer, hidden 128, dropout 0.3, AdamW + cosine LR, optional FastText init.
- **LogReg**: TF-IDF (uni+bigrams, 40K vocab, sublinear_tf), 5-fold GridSearchCV over `C`, balanced class weights. **3 seeds × 5 folds = 15 runs.**
- **SVM (LinearSVC + CalibratedCV)**: same TF-IDF, same CV. **Same 15 runs.**

### F5. Existing evaluation (verified, partial)

| Metric | Status |
|---|---|
| Accuracy, Precision, Recall, F1, ROC-AUC, AP | ✅ implemented (`metrics.py`) |
| Confusion matrix + grid + ROC + PR curves | ✅ |
| 5-fold stratified CV × 3 seeds | ✅ (LR/SVM only) |
| Bootstrap 95% CI (10,000 iter) | ✅ `bootstrap_confidence_interval` |
| McNemar + Holm-Bonferroni | ✅ |
| Cohen's d effect size | ✅ |
| Calibration (ECE, MCE, Brier, reliability diagram) | ✅ |
| Post-hoc calibration (Platt/Temp/Isotonic) | ✅ |
| Token-level attribution (SHAP, IG, attention) | ✅ all 4 models |
| Cross-model attribution agreement (faithfulness, rank) | ✅ |
| Knowledge distillation (PhoBERT → Student BiLSTM) | ✅ |
| Hard-cases analysis | ✅ |
| Ablation (TF-IDF vocab, n-gram, segmentation, C, sublinear_tf) | ✅ LR-only |

| Metric | Status |
|---|---|
| Temporal / time-based split | ❌ |
| Cross-domain / cross-topic split | ❌ |
| Adversarial / paraphrase robustness | ❌ |
| Multi-seed for deep models (BiLSTM, PhoBERT) | ❌ (single seed) |
| Per-class PR curves | ❌ (only AP) |
| Human evaluation / IAA on labels | ❌ |
| Claim-level evaluation | ❌ |
| Robustness under style perturbation | ❌ |
| Statistical comparison of ablation deltas | ❌ |

### F6. Engineering quality (verified)

- 17 test files; deterministic seed; central config (`config.py`); ExperimentTracker logs config hash + git commit.
- Clean OOP (one trainer per model, shared `runner.py`).
- README.md itself acknowledges: **"Dataset source/licence: UNKNOWN — see the thesis for source attribution."**

---

# Part A — Current System Audit (consolidated)

### What is strong

1. Engineering execution is **well above** typical student-project quality: OOP, tests, CLI, config centralisation, logging, calibration, attribution.
2. The 4-model comparison (LR/SVM/BiLSTM/PhoBERT) with statistical tests is publishable as a **benchmarking workshop paper**.
3. Knowledge-distillation ablation is a nice engineering demonstration.
4. Token-level attribution analysis (SHAP, IG, attention) is correctly implemented for all 4 models.

### What is weak

1. **Dataset provenance is unknown.** The README admits this. A thesis cannot defend results on an unverified dataset.
2. **The dataset is structurally heterogeneous** (formal news vs. social media). The model can exploit style rather than veracity. This is **the most important empirical finding** of this audit.
3. **Deep models are evaluated with a single seed (42).** No variance estimates. McNemar's test on one run per model is not strictly valid.
4. **No temporal / out-of-distribution evaluation.** The split is purely random.
5. **No adversarial robustness evaluation.**
6. **No ablation on the deep models.** Only LR is ablated.
7. **The architecture is text-only and monolithic** — `AutoModelForSequenceClassification` is the entire "architecture". This is the central limitation that a research thesis can attack.

### Score (0–10)

| Dimension | Score |
|---|---|
| Problem formulation | 6/10 |
| Dataset quality | 5/10 |
| NLP methodology | 7/10 |
| Model sophistication | 6.5/10 |
| Experimental methodology | 6/10 (single-seed deep, no OOD) |
| Evaluation quality | 7/10 |
| Software engineering | 8/10 |
| System architecture | 6/10 |
| Reproducibility | 7/10 |
| Academic contribution | 5.5/10 (currently benchmarking, not research) |

**Average: 6.5/10**. Class: **strong capstone / borderline thesis-ready**, requires one new research axis.

---

# Part B — Genuine Research Gaps

> "What is fundamentally missing or weak in the current formulation?"

### Gap A — Representation limitation

Current pipeline: **article → text → tokenizer → transformer → logit**. This is a "bag of evidence in a flat text" representation.

What is missing:

- **Claim-level structure**. Articles contain multiple verifiable claims. Modern fact-checking (FEVER, WiCE, ClaimDecomp) operates on claim–evidence pairs.
- **Sentence-level structure**. Articles contain multi-sentence arguments; some sentences are factual, others are emotive or speculative.
- **Source / provenance signals**. The dataset does not have these fields, but **stylometric proxies** (formal news vs. social media register) are observable.
- **Entity awareness**. Entities (PER, ORG, LOC, EVENT) carry credibility signals — fake news tends to misuse entities or attribute false claims to them.
- **Discourse structure**. The classical formal-news article structure (lead → context → quote → attribution) vs. social-media post structure (claim → emoji → hashtag → URL) is a structural cue.

### Gap B — Information limitation

The current model uses **only the text column**. The dataset does have:

- `id` — sequential ID, may carry ordering signal but not verified.
- `date` — only 532 parseable, but enough for a pilot study.
- **Implied source / register** (formal news vs. social-media) — derivable from text patterns.

It does **NOT** have external evidence, source metadata, or author identity. Retrieval-augmented verification is technically possible but requires building a new infrastructure (Vietnamese trusted corpus + NLI).

### Gap C — Reasoning limitation

The current classifier learns **"this text looks like fake news"** rather than **"these claims conflict with available evidence"**. There is no explicit reasoning mechanism.

However: a transformer **does** perform implicit reasoning over tokens. The question is whether to make this reasoning **explicit** (claim decomposition, evidence retrieval, evidence aggregation).

### Gap D — Generalisation limitation

Unverified but **highly likely**, given F2:

- The model may fail on new topics because its "fake" class is correlated with social-media style.
- It will fail on paraphrased fake news (because URL/mention/style cues are absent).
- It will fail on temporally distant news (because the formal-news vs. social-media mix shifts).

### Gap E — Shortcut learning

This is the **largest empirical finding** of the audit. The dataset is partially a **register classification problem**, not a credibility classification problem. The model can achieve high accuracy by learning:

- "ALL-CAPS + URL + @mention" → fake
- "Formal news attribution markers" → real

This is a real, measurable, publishable finding.

### Gap F — Single-seed deep-model evaluation

BiLSTM and PhoBERT are evaluated with one run. No variance. No statistical power. Cannot claim significance with confidence.

### Gap G — Dataset provenance

Unknown. Acknowledged in README.

---

# Part C — Candidate Research Directions

> 5 fundamentally different directions. Each must answer: what is the central mechanism, what is the research hypothesis, what is novel?

## Direction 1 — **Hierarchical Claim-Aware Vietnamese Fake News Detector**

### Research idea

Replace flat-text classification with **explicit claim-level reasoning**. Decompose the article into verifiable claims, encode each, then aggregate to a document-level decision. The **mechanism** is the hierarchical reasoning path: claim extraction → claim encoding → claim aggregation. This forces the model to reason over *what the article says*, not *what it looks like*.

### Problem

The current model compresses an article into a single `[CLS]` representation. This loses the multi-claim structure of news. When the article contains one true claim and one false claim, the model must average them and hope for the best.

### Hypothesis

A hierarchical model that explicitly reasons at the **claim level** will (a) produce more interpretable predictions, (b) be more robust under paraphrase (because claims are preserved when style is changed), and (c) achieve ≥1 F1 point gain over flat PhoBERT on a paraphrase-robust Vietnamese test set.

### Architecture (high-level)

```
Article (Vietnamese)
   │
   ├─► Sentence Splitter (heuristic)
   │
   ├─► Claim Extractor (PhoBERT-NER-style head, fine-tuned)
   │       outputs: list of claim spans
   │
   ├─► Claim Encoder (PhoBERT, shared weights)
   │       outputs: per-claim embeddings h_1, ..., h_n
   │
   ├─► Claim Aggregator (Transformer-Encoder over claim embeddings
   │       OR attention pooling OR Set-Transformer)
   │       outputs: document representation z
   │
   └─► Classifier Head (Linear → softmax)
            outputs: P(real) / P(fake)
```

### Novel component

The **claim-aggregator** is the genuinely new piece. Existing work uses flat `[CLS]`; here we introduce a learnable second-level encoder over claims. The novelty is **mechanism** (hierarchical reasoning), not just stacking layers.

### Data requirement

- Existing 15,789 Vietnamese articles.
- **Heuristic claim extraction** (split on punctuation, take sentences with strong claim verbs "khẳng định", "cho biết", "tuyên bố", etc.). No human annotation needed.
- **Synthetic claim labels**: take each sentence, label it as "claim" if it contains a factive verb, otherwise "non-claim". Or simply treat every sentence as a claim. This is a methodological choice to be tuned.

### Training objective

```
L_total = λ1 * L_classification(doc)         # cross-entropy on doc label
       + λ2 * L_claim_regularization          # auxiliary: encourage diverse claim embeddings
```

λ1 = 1.0, λ2 = 0.1 (small auxiliary). Optional.

### Experimental validation

- Compare against flat PhoBERT (existing).
- Inspect attention weights over claims to verify the model focuses on informative claims.
- Compare on a paraphrased Vietnamese test set (back-translation).

### Ablation

- A1: Flat `[CLS]` (existing PhoBERT).
- A2: Mean-pool over sentence embeddings.
- A3: Hierarchical with attention pooling (proposed).
- A4: Hierarchical with Transformer-Encoder aggregator.
- A5: Hierarchical + claim-filtering (skip non-claim sentences).

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **Medium-High** (first explicit claim-aggregation for Vietnamese) |
| Technical depth | **High** (hierarchy, auxiliary losses) |
| Research value | **High** (mechanism is clear: hierarchical reasoning) |
| Feasibility | **Medium** (claim extraction needs care) |
| Dataset feasibility | **High** (no annotation needed) |
| Implementation complexity | **Medium** |
| Experimental complexity | **Medium** |
| Risk | **Medium** (claim extraction quality is the bottleneck) |
| Explainability | **Very High** (claim attention is human-readable) |
| Thesis defensibility | **High** |

**Overall: ~7.0/10.** Strong candidate, feasible in 5–6 months.

---

## Direction 2 — **Disentangled Style/Content Encoder for Robust Vietnamese Fake News Detection**

### Research idea

The dataset is **partially** a style-classification problem (F2). A research-grade model should **explicitly disentangle** style from content, then make decisions primarily from content. The **mechanism** is adversarial disentanglement: the style encoder should be unable to predict the label, forcing the content encoder to do the work.

### Problem

If a model can classify by style alone (formal news = real, social media = fake), it will fail catastrophically when (a) fake news is written in formal news register, or (b) real news is shared on social media. Both are common in practice.

### Hypothesis

A model trained with adversarial style-vs-content disentanglement will:

(a) achieve comparable in-domain F1 (≤0.5 point drop);
(b) gain ≥3 F1 points on style-shuffled test sets (content of one label + style of the other);
(c) gain ≥5 F1 points on out-of-distribution Vietnamese test sets.

### Architecture (high-level)

```
Article
   │
   ├─► Shared Encoder (PhoBERT, frozen or partially frozen)
   │       outputs: contextual embeddings H
   │
   ├─► Style Head: BiLSTM + projection → s
   │       adversarial loss: -L_label(s)  ← gradient reversal
   │
   ├─► Content Head: BiLSTM + projection → c
   │       prediction loss: L_label(c)
   │
   └─► Classifier on c (and optionally c + mean(H))
```

### Novel component

The **adversarial style-content disentanglement** is the contribution. The model is forced to learn content features that ignore style, which is hypothesised to be more robust.

### Data requirement

- Existing 15,789 Vietnamese articles.
- **Style features** (deterministically extracted): punctuation density, exclamation count, ALL-CAPS ratio, sentence length distribution, attribution-marker count.
- **Style-perturbation augmentation**: paraphrasing, all-caps injection, social-media-style re-writing.

### Training objective

```
L_total = L_classification(c) + γ * L_style_aux      # style prediction via auxiliary classifier on s
       - α * L_style_adv                                # gradient reversal on style classifier
```

α is the adversarial weight (start 0.1, ramp to 1.0). γ is the auxiliary style-prediction loss (0.1).

### Experimental validation

- **Style-shuffled test**: take the content of a real article and the style of a fake article (or vice versa). If the model truly uses content, it should still predict correctly.
- **Style-perturbation test**: paraphrase, all-caps, emoji injection.
- **Out-of-distribution test**: held-out topic or domain (if available).

### Ablation

- B1: PhoBERT (existing flat) — **strong baseline**.
- B2: PhoBERT + style classifier only (no disentanglement).
- B3: PhoBERT + adversarial disentanglement.
- B4: B3 + style-augmented training (style-perturbation in training set).

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **High** (first explicit disentanglement for Vietnamese fake news) |
| Technical depth | **High** (adversarial training, gradient reversal, multi-loss) |
| Research value | **High** (directly addresses F2) |
| Feasibility | **Medium-High** (existing tools: `torch.autograd.grad_reverse`) |
| Dataset feasibility | **High** |
| Implementation complexity | **Medium** |
| Experimental complexity | **Medium-High** (need style-perturbation pipeline) |
| Risk | **Medium** (disentanglement rarely works perfectly) |
| Explainability | **Medium** |
| Thesis defensibility | **High** |

**Overall: ~7.5/10.** Strongest candidate on novelty × mechanism clarity.

---

## Direction 3 — **Multi-View Evidence Fusion: Text + Linguistic Features + Stylometry**

### Research idea

Treat the article as a **multi-view object**: (1) semantic text view (transformer embedding), (2) linguistic feature view (POS, NER, dependency parses), (3) stylometric view (register, punctuation, length). Each view is encoded separately, then **fused** via cross-attention. The **mechanism** is explicit multi-view fusion — each view captures orthogonal information, and the fusion learns to weight them per-instance.

### Problem

The current flat-text classifier weights all information equally. A short social-media post with no entities is treated identically to a long formal news article with 30 entities. Per-instance weighting is missing.

### Hypothesis

A multi-view model with cross-view attention will (a) match flat PhoBERT on in-domain test, (b) gain ≥1.5 F1 on out-of-distribution Vietnamese test, and (c) produce per-instance explainable views.

### Architecture

```
Article
   │
   ├─► View 1: PhoBERT → semantic embeddings H_s
   │
   ├─► View 2: Linguistic features → H_l (NER tags, POS tags, claim verbs)
   │             encoded via small BiLSTM
   │
   ├─► View 3: Stylometric features → H_st (length, caps, exclam, URL count)
   │             encoded via MLP
   │
   ├─► Cross-View Fusion (Transformer with H_s, H_l, H_st as input)
   │       outputs: fused representation F
   │
   └─► Classifier on F
```

### Novel component

**Cross-view attention over heterogeneous views** is the contribution. Each view is a sequence of tokens; the transformer learns per-instance which view to attend to.

### Data requirement

- Existing 15,789 articles.
- Linguistic features: PhoBERT-NER (Vietnamese), `underthesea` POS, claim-verb lexicon.
- Stylometric features: deterministic (F2 already measured these).

### Training objective

```
L_total = L_classification(F)
```

Plus optional: view-specific auxiliary losses for self-supervision of each view.

### Experimental validation

- Compare against flat PhoBERT.
- Visualise cross-view attention to verify per-instance view selection.
- Test on style-shuffled set (forces view 1 to dominate) and entity-scrambled set (forces view 2).

### Ablation

- C1: PhoBERT (flat).
- C2: PhoBERT + stylometry concatenated.
- C3: PhoBERT + linguistic concatenated.
- C4: PhoBERT + both concatenated.
- C5: Multi-view cross-attention (proposed).

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **Medium** (multi-view is a known technique) |
| Technical depth | **High** (heterogeneous fusion is non-trivial) |
| Research value | **Medium** (less mechanism clarity than #2) |
| Feasibility | **Medium** (requires NER pipeline) |
| Dataset feasibility | **High** |
| Implementation complexity | **Medium-High** |
| Experimental complexity | **High** |
| Risk | **Medium-High** (NER quality on noisy text) |
| Explainability | **High** |
| Thesis defensibility | **Medium-High** |

**Overall: ~6.5/10.** Solid but not as mechanism-clear as #2.

---

## Direction 4 — **Contrastive Robustness Learning for Vietnamese Fake News**

### Research idea

Define **positive pairs** (same article, style-perturbed: e.g., paraphrased) and **negative pairs** (articles of different labels). Train the encoder with a contrastive objective **in addition** to the classification loss. The **mechanism** is that the encoder learns to be invariant to style perturbations while remaining sensitive to content.

### Problem

The current encoder is sensitive to style perturbations (URL, hashtags, all-caps). Real-world fake news does not always carry these cues.

### Hypothesis

A contrastively-trained encoder will lose ≤1 F1 point on in-domain test but gain ≥3 F1 points on style-perturbed Vietnamese test (paraphrase, synonym swap, all-caps).

### Architecture

```
Article
   │
   ├─► PhoBERT encoder f
   │
   ├─► Projection head g (2-layer MLP) → embedding e
   │
   └─► Classification head h
```

**Two losses**:

```
L_cls  = cross-entropy(h(f(x)), y)
L_cont = -log [ exp(sim(e, e+)) / Σ exp(sim(e, e-)) ]
```

L_total = L_cls + β * L_cont, β tuned (start 0.1).

**Augmentations for positive pairs**:
- back-translation (vi→en→vi)
- synonym swap (PhoBERT-based)
- character-level noise (homoglyph, swap, delete)
- emoji injection / removal

### Novel component

The **style-perturbation augmentation** is the contribution; the contrastive framework itself is standard. The novelty lies in demonstrating that style-invariance improves robustness on **Vietnamese** fake-news detection.

### Data requirement

- Existing 15,789 articles.
- Augmentation pipeline (back-translation requires an English-Vietnamese model — verify availability; character-level noise is trivial).

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **Medium-High** (style-invariant contrastive for Vietnamese) |
| Technical depth | **Medium-High** |
| Research value | **High** (robustness is publishable) |
| Feasibility | **High** (algorithm is standard) |
| Dataset feasibility | **High** |
| Implementation complexity | **Medium** |
| Experimental complexity | **Medium** |
| Risk | **Medium** (back-translation quality) |
| Explainability | **Low-Medium** |
| Thesis defensibility | **Medium-High** |

**Overall: ~6.5/10.** Solid engineering + research combination.

---

## Direction 5 — **Multi-Task Learning: Fake News + Style/Register Prediction**

### Research idea

Train the encoder **jointly** on (a) fake-news classification (main task) and (b) style/register prediction (auxiliary task). The auxiliary task provides **inductive bias**: the encoder learns to separate style from content because the auxiliary task explicitly requires style identification. The classification head then has access to a representation where style is well-modelled and can be discounted.

### Problem

Without an explicit signal, the model may conflate style with credibility.

### Hypothesis

A multi-task encoder where the auxiliary task is **style/register prediction** will produce a cleaner content representation, yielding ≥1 F1 point gain on style-shuffled Vietnamese test.

### Architecture

```
Shared PhoBERT encoder f
   │
   ├─► Head A: classification (fake/real)
   │
   └─► Head B: style/register classification (formal news / social media / mixed)
```

```
L_total = L_A + δ * L_B
```

δ is small (e.g. 0.3) to avoid style-overfitting.

### Novel component

The **explicit auxiliary style task** is the contribution. The intuition is: "by forcing the encoder to predict style, we expose style features; by then classifying credibility from the encoder, we learn to discount style."

### Data requirement

- Existing 15,789 articles.
- **Synthetic style labels**: classify each article as `formal_news` (has attribution markers, ≥30 words, low exclamation) or `social_media` (has URL/mention/hashtag or high exclamation). Hybrid `mixed`.

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **Medium** (multi-task is well-known) |
| Technical depth | **Medium** |
| Research value | **Medium** |
| Feasibility | **High** |
| Dataset feasibility | **High** |
| Implementation complexity | **Low-Medium** |
| Experimental complexity | **Medium** |
| Risk | **Low-Medium** |
| Explainability | **Medium** |
| Thesis defensibility | **Medium** |

**Overall: ~6.0/10.** Solid but the mechanism (inductive bias from auxiliary task) is weaker than #2.

---

## Direction 6 — **Claim Decomposition + Stance Reasoning (Evidence-Aware Lite)**

### Research idea

Decompose the article into claims. For each claim, predict a **stance** with respect to the article's overall label: "this claim supports the fake label", "this claim supports the real label", or "this claim is neutral". Aggregate per-claim stances to a document decision.

### Problem

Flat classification does not allow per-claim inspection. A claim-level decomposition makes the model auditable.

### Hypothesis

A claim-level stance aggregator matches flat PhoBERT in F1 but provides auditable explanations. The contribution is **explainability**, not raw accuracy.

### Architecture

```
Article
   │
   ├─► Claim extractor (heuristic + PhoBERT-filtered)
   │
   ├─► PhoBERT-stance-head: predict stance ∈ {supports_real, supports_fake, neutral} per claim
   │
   └─► Aggregator: weighted sum → softmax over {real, fake}
```

### Novel component

The **stance-aware aggregator** is the contribution; the explainability benefit is genuine.

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **Medium** |
| Technical depth | **Medium** |
| Research value | **Medium** (explainability focus) |
| Feasibility | **High** |
| Implementation complexity | **Medium** |
| Experimental complexity | **Medium** |
| Risk | **Low-Medium** |
| Explainability | **Very High** |
| Thesis defensibility | **Medium-High** |

**Overall: ~6.0/10.** Strong on explainability but weak on accuracy contribution.

---

## Direction 7 — **Retrieval-Augmented Verification (RAG + NLI)** [expensive]

### Research idea

For each article, retrieve top-k evidence from a Vietnamese trusted corpus (VnExpress archive, BBC Vietnamese, official RSS). Run NLI on (claim, evidence) pairs. Aggregate.

### Verdict

**High research value, high feasibility risk.** Requires building a Vietnamese trusted corpus (≥100K articles, scraping agreements) and a Vietnamese NLI model (Vietnamese NLI datasets are scarce). 5–6 months is tight. **Possible but ambitious.** If done well, this is the strongest possible thesis contribution. If retrieval fails or NLI is weak, the thesis fails.

### Scoring

| Criterion | Score |
|---|---|
| Novelty | **Very High** |
| Technical depth | **Very High** |
| Research value | **Very High** |
| Feasibility | **Medium-Low** |
| Dataset feasibility | **Low-Medium** (need new corpus) |
| Implementation complexity | **High** |
| Experimental complexity | **High** |
| Risk | **High** |
| Thesis defensibility | **High (if works) / Low (if retrieval fails)** |

**Overall: ~7.5/10 if executed, ~4/10 if not.** The variance is large.

---

## Direction 8 — **Temporal Adaptation via Date-Aware Training** [limited feasibility]

### Research idea

Recover the 532 dated samples, train with date conditioning, evaluate temporal generalisation.

### Verdict

**Medium novelty but low feasibility** because only 532 dates are parseable. Insufficient for a thesis-grade temporal study.

### Scoring

| Criterion | Score |
|---|---|
| Novelty | Medium |
| Feasibility | **Low** (data) |
| Risk | High |
| **Overall** | ~4/10 |

---

# Part D — Top 3 Architectures (Detailed)

> Selected based on the scoring: **mechanism clarity > novelty > feasibility > risk**.

### 🥇 Direction 2 — Disentangled Style/Content Encoder (highest mechanism clarity, directly addresses F2)

### 🥈 Direction 1 — Hierarchical Claim-Aware Encoder (strong mechanism, novel for Vietnamese)

### 🥉 Direction 3 — Multi-View Fusion (solid engineering, useful as a comparison)

---

## D.1 — Disentangled Style/Content Encoder (DETAILED)

### Inputs

- Article text (Vietnamese, raw or cleaned)
- Style feature vector: `[url_count, hashtag_count, mention_count, caps_ratio, exclam_count, avg_word_len, n_sentences, attribution_markers, sentence_len_std]` (all deterministic)

### Encoders

- **Shared encoder**: PhoBERT (`vinai/phobert-base`), all layers trainable, max_seq_len=256.
- **Style head**: BiLSTM (1 layer, hidden=128) over `H` (mean-pooled) → projection (128 → 64) → `s`.
- **Content head**: BiLSTM (1 layer, hidden=128) over `H` → projection (128 → 64) → `c`.
- **Classifier**: MLP (64 → 32 → 2) on `c`.

### Loss function

```
L_class = cross_entropy(MLP(c), y)               # main task
L_style = cross_entropy(style_classifier(s), y_style)   # auxiliary style prediction

# Gradient reversal layer multiplies L_style gradient by -alpha
L_total = L_class + γ * L_style
       = L_class + γ * (-α) * gradient_style_branch
```

Equivalently, using a Gradient Reversal Layer (GRL):

```
s_detached = GRL(c)        # gradient flips sign when flowing back
L_aux     = cross_entropy(MLP(s_detached), y_style)
L_total   = L_class + λ * L_aux
```

where λ starts at 0.1 and is ramped to 1.0 over training.

**Mechanism**: the gradient reversal forces the content branch `c` to be **uninformative about style**. The classifier must therefore rely on `c` to predict the label, which means `c` must contain label-relevant (credibility) information but NOT style information.

### Training procedure

- Optimiser: AdamW (encoder: lr=3e-5, heads: lr=1e-3)
- Warmup: 10% of total steps
- Batch size: 16 × grad accum 4 (effective 64)
- Epochs: 8, patience 2
- Multi-seed: 5 seeds (NEW, replaces single-seed baseline)
- λ ramp: 0.1 → 1.0 over 4 epochs

### Pseudocode (PyTorch skeleton)

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DisentangledClassifier(nn.Module):
    def __init__(self, phobert_name="vinai/phobert-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(phobert_name)
        h = self.encoder.config.hidden_size
        self.style_head  = nn.Sequential(nn.LSTM(h, 128, batch_first=True, bidirectional=True),
                                          nn.Linear(256, 64))
        self.content_head = nn.Sequential(nn.LSTM(h, 128, batch_first=True, bidirectional=True),
                                           nn.Linear(256, 64))
        self.cls_head     = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.style_cls    = nn.Linear(64, 3)  # formal / social / mixed

    def forward(self, input_ids, attention_mask, alpha=1.0):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Mean pool
        mask = attention_mask.unsqueeze(-1).float()
        h_pool = (H * mask).sum(1) / mask.sum(1).clamp(min=1)
        s = self.style_head(h_pool)
        c = self.content_head(h_pool)
        logits = self.cls_head(c)
        s_rev = GradientReversalFunction.apply(s, alpha)
        style_logits = self.style_cls(s_rev)
        return logits, style_logits
```

### Loss equation (final form)

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(\text{MLP}(c), y) + \lambda \cdot \mathcal{L}_{CE}(\text{GRL}(c), y_{style})$$

with $\lambda$ ramped from 0.1 to 1.0.

### Why this is research-grade

- The mechanism is **explicit**: adversarial gradient reversal.
- The contribution is **architectural + methodological**: the model is forced to learn style-invariant credibility.
- The evaluation is **clean**: style-shuffled test sets are a novel evaluation protocol for Vietnamese.

---

## D.2 — Hierarchical Claim-Aware Encoder (DETAILED)

### Inputs

- Article text (Vietnamese)
- Heuristically extracted claims (sentences containing factive/assertive verbs)

### Encoders

- **Claim extractor**: PhoBERT with a binary head per token (claim / non-claim) OR a heuristic sentence-level filter.
- **Claim encoder**: PhoBERT (shared weights), max_seq_len=128 per claim.
- **Claim aggregator**: 2-layer Transformer-Encoder over claim embeddings → document representation.

### Loss

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(\text{agg}(H_{claims}), y) + \gamma \cdot \mathcal{L}_{aux}$$

where $\mathcal{L}_{aux}$ is an auxiliary loss on claim embeddings (e.g. diversity regularisation).

### Architecture diagram

```
Article
   │
   ├─► Sentence Splitter (punctuation-based, Vietnamese-aware)
   │
   ├─► Claim Filter (sentence contains assertive verb OR NER entities)
   │       outputs: list of claim texts C = {c_1, ..., c_n}
   │
   ├─► PhoBERT (shared) → {h_1, ..., h_n}
   │
   ├─► Claim Aggregator (2-layer Transformer-Encoder)
   │       outputs: z = mean(h_agg)
   │
   └─► Classifier (Linear(z) → softmax)
```

### Why this is research-grade

- Hierarchical reasoning is a documented mechanism (long-document classification, evidence aggregation).
- The claim-level output is **interpretable**: you can show "the model based its decision on claims #3, #7".
- For Vietnamese specifically, this is novel.

---

## D.3 — Multi-View Fusion (DETAILED)

### Inputs

- Article text
- Linguistic features (NER tags via `underthesea`, POS tags, claim-verb count)
- Stylometric features (length, caps, exclam, URL count, attribution markers)

### Encoders

- View 1 (semantic): PhoBERT → H_s
- View 2 (linguistic): BiLSTM over NER/POS sequences → H_l
- View 3 (stylometric): MLP over feature vector → H_st

### Cross-view fusion

- Treat the three view representations as input to a 3-token Transformer-Encoder.
- Output: fused representation F.

### Loss

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(\text{MLP}(F), y)$$

### Why this is research-grade (less so)

- Multi-view is a known technique; novelty is lower.
- But it provides a useful comparison to disentanglement (#2).

---

# Part E — Recommended Architecture

## ✅ Recommendation: **Direction 2 — Disentangled Style/Content Encoder** as primary, with **Direction 1 (Hierarchical)** as a strong comparative chapter.

### Why this is the best thesis direction

#### 1. It directly addresses the **most important empirical finding** (F2)

The dataset is partially a register-classification problem. The current model is incentivised to exploit this. A disentanglement architecture **explicitly** forces the model to use content, not style. This is **mechanism-driven**, not engineering-driven.

#### 2. The mechanism is **defensible**

Adversarial gradient reversal is a known technique (Ganin et al., 2016) but its application to Vietnamese fake-news detection is novel. The student can:

- Explain the gradient reversal mathematically.
- Justify why adversarial disentanglement should work (style invariance ↔ robust credibility).
- Defend the choice of style auxiliary task.

This is what examiners look for.

#### 3. The evaluation protocol is **novel**

A **style-shuffled test set** (content of label A + style of label B) is a new evaluation methodology for Vietnamese fake-news detection. This alone is a publishable contribution.

#### 4. It is **feasible within 5–6 months**

- Architecture: ~2 weeks to implement.
- Training: 5 seeds × 8 epochs × 2 hours/epoch on a single GPU = 3 days wall time.
- Style-shuffled test set generation: ~1 week.
- Statistically rigorous comparison: existing tools (`statistical_tests.py`).

#### 5. The contribution is **multi-dimensional**

- Architectural: novel style-content disentanglement.
- Methodological: style-shuffled test set protocol.
- Empirical: first quantification of register-bias in Vietnamese fake-news detection.
- Engineering: reusable `GradientReversalLayer` for Vietnamese NLP.

#### 6. The combination with Direction 1 (Hierarchical) gives **two chapters of novel contribution**

- Chapter 4: Style-shuffled test protocol + style bias quantification (Direction 2 part 1).
- Chapter 5: Disentangled architecture (Direction 2 part 2).
- Chapter 6: Hierarchical claim-aware architecture (Direction 1).
- Chapter 7: Combined architecture (Direction 2 + Direction 1) — claims-level disentanglement.

This is four chapters of genuine contribution. Easily thesis-defensible.

---

# Part F — Novelty Statement (Conservative)

> **This thesis investigates whether current Vietnamese fake-news classifiers exploit surface-style cues (URLs, hashtags, exclamation marks, register) rather than semantic credibility. We propose a Disentangled Style/Content Encoder that explicitly separates style and content via adversarial gradient reversal, and evaluate it on a novel Style-Shuffled Vietnamese Test Set that isolates content from style. We further propose a Hierarchical Claim-Aware Encoder that decomposes articles into claims and aggregates them via a second-level transformer. The contribution is architectural (the disentanglement + hierarchical aggregation), methodological (the style-shuffled protocol), and empirical (the first quantification of register-bias in Vietnamese fake-news detection).**

### Honest novelty classification

| Component | Novelty |
|---|---|
| Adversarial disentanglement (Ganin et al., 2016) | **Existing idea**, well-known in domain adaptation |
| Style-shuffled test set | **Methodological novelty** for Vietnamese |
| Disentanglement applied to Vietnamese fake-news | **Meaningful adaptation** |
| Hierarchical claim-aware encoder | **Existing idea** (long-doc classification) **adapted meaningfully** to Vietnamese |
| Combined claim-level disentanglement | **Architectural novelty** (combining two mechanisms is novel for this task) |
| Empirical register-bias quantification | **Empirical novelty** for Vietnamese |

**Do NOT claim** "first architecture to disentangle style from content" — that would be false (MIXOUT, style transfer, etc. exist). **DO claim** "first application of adversarial style-content disentanglement to Vietnamese fake-news detection, with a novel style-shuffled evaluation protocol that isolates content credibility from surface-style cues."

This is honest, defensible, and publishable.

---

# Part G — Dataset Schema (Required by Recommended Architecture)

### Original fields (already exist)

| Field | Type | Source | Use |
|---|---|---|---|
| `id` | int | raw.csv | sample identifier |
| `text` | string | raw.csv | input to model |
| `date` | string (mostly NaT) | raw.csv | auxiliary (96.6% missing); cannot drive temporal study |
| `label` | int (0/1) | raw.csv | target |

### Deterministically derived fields (NEW — required by the architecture)

| Field | Definition | Source | Cost | Reliability | Leakage Risk | Use |
|---|---|---|---|---|---|---|
| `url_count` | count of `http(s)://` or `www.` patterns | deterministic | zero | exact | **none** (URLs are removed in preprocessing but the count is preserved) | style head input |
| `hashtag_count` | count of `#\w+` patterns | deterministic | zero | exact | none | style head input |
| `mention_count` | count of `@\w+` patterns | deterministic | zero | exact | none | style head input |
| `exclamation_count` | count of `!` | deterministic | zero | exact | none | style head input |
| `caps_ratio` | upper-case chars / total chars | deterministic | zero | exact | none | style head input |
| `avg_word_length` | mean word length | deterministic | zero | exact | none | style head input |
| `n_sentences` | count of sentence-ending punctuation | deterministic | zero | exact | none | style head input |
| `attribution_markers` | count of "Theo", "TTXVN", "Bộ Y tế", "Reuters", etc. | regex | zero | exact | **MEDIUM** — if attribution is itself a label cue, removing the cue from the model is desirable | style head input |
| `punctuation_density` | punctuation chars / total chars | deterministic | zero | exact | none | style head input |

### NLP-derived fields (NEW — used by Hierarchical / Claim decomposition)

| Field | Definition | Method | Cost | Reliability | Leakage Risk | Use |
|---|---|---|---|---|---|---|
| `claim_sentences` | list of sentence indices that are claims | regex on assertive/factive verbs (Vietnamese lexicon: "khẳng định", "tuyên bố", "cho biết", "nói", "theo", "thông báo") | zero (heuristic) | approximate | low | claim extractor input |
| `entities` | list of (text, type) tuples | `underthesea` NER or PhoBERT-NER | 30 min for 15K | 70-85% F1 | low | linguistic view (Direction 3) |
| `pos_tags` | per-token POS | `underthesea` | 30 min | high | none | linguistic view |

### Style auxiliary labels (NEW — for adversarial disentanglement only)

| Field | Definition | Method | Cost | Reliability | Leakage Risk | Use |
|---|---|---|---|---|---|---|
| `style_label` | {0: formal_news, 1: social_media, 2: mixed} | deterministic rule (has attribution + ≥30 words + low exclam → 0; has URL/hashtag or high exclam → 1; else 2) | zero | high (rule-based) | **medium** — the style label is correlated with the credibility label | auxiliary task for adversarial disentanglement |

**Important caveat**: `style_label` is correlated with `label` (F2). This is the **point** of the auxiliary task: we want the content encoder `c` to NOT be able to predict `style_label`, because the only way to make `c` invariant to style is to push style information into the discarded branch. The correlation is intentional.

### Externally retrieved fields

**None required** for the recommended architecture. Retrieval-augmented verification (Direction 7) is excluded due to time budget.

### Human annotations

**None required.** All new fields are deterministically derived or heuristic. The thesis can defend "no additional human labelling needed".

### Dataset size

- Existing: 15,789 raw → 13,958 cleaned → 13,958 (after preprocessing).
- **No expansion needed** for the recommended architecture.

---

# Part H — Research Questions (precise)

### RQ1 — Empirical baseline

> **RQ1**: To what extent does the existing PhoBERT Vietnamese fake-news classifier exploit surface-style cues (URLs, hashtags, all-caps, register) rather than semantic credibility?

*Why*: this is the empirical finding that motivates the thesis.

### RQ2 — Mechanism evaluation

> **RQ2**: Does an adversarially-disentangled style/content encoder (Direction 2) improve robustness on a novel style-shuffled Vietnamese test set, compared with flat PhoBERT?

*Why*: tests the central architectural hypothesis.

### RQ3 — Hierarchical reasoning

> **RQ3**: Does a hierarchical claim-aware encoder (Direction 1) provide complementary gains beyond style disentanglement, especially on paraphrase-perturbed Vietnamese test samples?

*Why*: tests whether explicit claim-level reasoning adds value.

### RQ4 — Combination

> **RQ4**: Does the combined architecture (claim-level disentanglement) outperform either component alone?

*Why*: tests compositional contribution.

### RQ5 — Multi-seed reproducibility

> **RQ5**: Are the reported gains of the proposed architectures stable across ≥5 random seeds, with statistical significance (McNemar + Holm-Bonferroni)?

*Why*: ensures the contribution is not a single-seed artefact.

### RQ6 — Calibration and interpretability

> **RQ6**: Does the disentangled architecture produce more interpretable attributions (claim-level attributions) than flat PhoBERT, and does it maintain comparable calibration (ECE)?

*Why*: secondary contribution on explainability.

---

# Part I — Testable Hypotheses

### H1 (RQ1)

> PhoBERT Vietnamese fake-news classifier achieves ≥85% F1 on in-domain test but drops to ≤70% F1 on a **style-shuffled test set** where content of one label is paired with style of the other.

*How to test*: train flat PhoBERT; build style-shuffled test set by swapping style features; measure F1.

### H2 (RQ2)

> The Disentangled Style/Content Encoder (Direction 2) recovers ≥50% of the style-shuffled F1 drop while losing ≤1 F1 point on in-domain test.

*How to test*: train Direction 2 architecture with 5 seeds; compare in-domain and style-shuffled F1 vs flat PhoBERT.

### H3 (RQ3)

> The Hierarchical Claim-Aware Encoder (Direction 1) gains ≥1 F1 point on a paraphrase-perturbed Vietnamese test set vs flat PhoBERT.

*How to test*: build paraphrase test set via back-translation (vi→en→vi); compare F1.

### H4 (RQ4)

> The combined architecture (claim-level disentanglement) outperforms either component alone by ≥0.5 F1 point on the style-shuffled test, with p<0.05 (paired McNemar + Holm-Bonferroni).

### H5 (RQ5)

> All reported gains are stable across 5 seeds with effect size Cohen's d ≥ 0.5.

### H6 (RQ6)

> The disentangled encoder maintains ECE ≤ 0.05 on in-domain test (matching flat PhoBERT).

---

# Part J — Experiment Matrix

## Baselines

| ID | Model | Description | Source |
|---|---|---|---|
| **B0** | TF-IDF + LogReg | existing | repo |
| **B1** | TF-IDF + SVM (linear) | existing | repo |
| **B2** | BiLSTM + FastText | existing | repo |
| **B3** | PhoBERT (flat) | existing, **multi-seed** (NEW) | repo |
| **B4** | PhoBERT + post-hoc calibration (Platt) | existing | repo |

## Proposed variants

| ID | Model | Description |
|---|---|---|
| **M1** | PhoBERT + style auxiliary head (no disentanglement) | baseline for disentanglement ablation |
| **M2** | Disentangled Style/Content Encoder (Direction 2) | gradient reversal, λ=0.1→1.0 |
| **M3** | Hierarchical Claim-Aware Encoder (Direction 1) | 2-layer Transformer-Encoder aggregator |
| **M4** | Combined: claim-level disentanglement | Direction 1 + Direction 2 |
| **M5** | M2 + style-augmented training (style-perturbation in train) | perturbation augmentation |

## Ablation

| Ablation | Description |
|---|---|
| A1: λ=0 (no adversarial loss) | tests gradient reversal necessity |
| A2: λ=1 (constant, no ramp) | tests ramp schedule |
| A3: Random style labels | tests whether adversarial loss is doing anything |
| A4: Heuristic claim extraction only (no NER) | isolates Direction 1 vs heuristic |
| A5: Mean-pool aggregator instead of Transformer-Encoder | tests aggregator design |
| A6: 1-layer vs 2-layer aggregator | tests depth |

## Robustness

| Test | Construction |
|---|---|
| Style-shuffled test | swap style features between label-A and label-B samples (1:1 pairing, keep content) |
| Paraphrase test | back-translate vi→en→vi using a Vietnamese MT model (verify availability) |
| Synonym-swap test | replace 20% of content words with PhoBERT-context-substitutes |
| All-caps test | inject ALL-CAPS into 30% of tokens |
| Emoji injection | add 1–3 random emojis at the end |
| URL masking | ensure URL count goes to 0 at test time |

## Generalisation

| Test | Construction |
|---|---|
| Source split | not possible (no source field) |
| Topic split | topic-cluster articles using a small Vietnamese zero-shot classifier; report per-topic F1 |
| Length split | short vs long articles; report per-length-bucket F1 |
| Time split | 532 dated samples split by date; report temporal F1 (small-scale) |

## Statistical validation

- 5 random seeds for each model.
- Bootstrap 95% CI on F1 (10,000 iterations, existing).
- Paired McNemar on per-sample predictions.
- Holm-Bonferroni correction for multiple comparisons.
- Cohen's d effect size (existing).

---

# Part K — 5–6 Month Roadmap

## Month 1 — Foundation & empirical baseline

**Weeks 1–2**: Reproduce baseline.
- Re-run all 4 baselines with 5 seeds (BiLSTM and PhoBERT were single-seed before — this is the fix).
- Lock results.
- Document dataset provenance (write Datasheet).
- Audit and re-run existing evaluation suite (cross-validation, calibration, attribution).

**Weeks 3–4**: Empirical register-bias quantification (RQ1).
- Implement deterministic style-feature extractor (all 9 features).
- Build style-shuffled test set.
- Measure PhoBERT F1 on style-shuffled set. **This is the empirical contribution of Chapter 4.**

**Deliverable**: Chapter 4 draft (RQ1 + H1 result).

## Month 2 — Architecture implementation (Direction 2)

**Weeks 5–6**: Implement Disentangled Style/Content Encoder (Direction 2).
- Gradient reversal layer.
- Style head + content head + classifier.
- Style auxiliary task labels (rule-based: formal_news / social_media / mixed).
- λ ramp schedule.

**Weeks 7–8**: Initial training + debugging.
- Train with 5 seeds.
- Compare with flat PhoBERT on in-domain test.
- Compare on style-shuffled test.

**Deliverable**: Chapter 5 draft (RQ2 + H2 result).

## Month 3 — Architecture implementation (Direction 1) + paraphrasing pipeline

**Weeks 9–10**: Implement Hierarchical Claim-Aware Encoder (Direction 1).
- Heuristic claim extractor (Vietnamese assertive verb lexicon).
- PhoBERT claim encoder (shared).
- Transformer-Encoder claim aggregator.

**Weeks 11–12**: Build paraphrase perturbation pipeline.
- Verify availability of a Vietnamese↔English MT model.
- Implement back-translation pipeline.
- Build paraphrase test set.

**Deliverable**: Chapter 6 draft (RQ3 + H3 result).

## Month 4 — Combination + robustness experiments

**Weeks 13–14**: Combined architecture (M4).
- Claim-level disentanglement.
- Train with 5 seeds.
- Compare with M2 and M3 alone.

**Weeks 15–16**: Robustness experiments.
- Synonym-swap test.
- All-caps injection.
- Emoji injection.
- Length-bucket analysis.

**Deliverable**: Chapter 7 draft (RQ4 + H4 result, plus robustness analysis).

## Month 5 — Statistical validation + ablation + writing

**Weeks 17–18**: Ablation + statistical analysis.
- Run A1–A6 ablations.
- Apply Holm-Bonferroni, McNemar, bootstrap CI, Cohen's d.
- Generate final tables and figures.

**Weeks 19–20**: Thesis writing — Chapter 1 (Introduction), Chapter 2 (Related Work), Chapter 3 (Dataset & Methods).

**Deliverable**: First draft of Chapters 1–3.

## Month 6 — Finalisation

**Weeks 21–22**: Thesis writing — Chapters 4–7 (results), Chapter 8 (Discussion), Chapter 9 (Conclusion).

**Weeks 23–24**: Polish, reproducibility checks, code release, final figures, appendices.

**Deliverable**: Final thesis.

### Risks by month

| Month | Risk | Fallback |
|---|---|---|
| 1 | Style-shuffled test shows <5 F1 drop → less motivation for disentanglement | Still publish RQ1 + write about register bias qualitatively |
| 2 | Adversarial disentanglement is unstable (loss does not converge) | Try L2-statistical distance instead of adversarial; or switch to Direction 1 |
| 3 | Vietnamese↔English MT not available for back-translation | Use synonym swap only (PhoBERT-context-substitute); report paraphrase test only with this |
| 4 | Combined architecture (M4) does not outperform M2 + M3 | Report M2 + M3 as separate contributions; combined is a stretch |
| 5–6 | Writing timeline slips | Submit thesis as-is with placeholder figures; refine later |

---

# Part L — Risk Analysis

## L.1 — Risks that could kill the project

### Risk 1 — Disentanglement fails to converge

**Probability**: Medium-High (30%).
**Impact**: High.
**Fallback**: Use a simpler style-removal baseline (strip style features from input, train flat PhoBERT) as the disentanglement approach. Compare against that.

### Risk 2 — The dataset is too noisy for clean separation

**Probability**: Medium (25%).
**Impact**: Medium.
**Fallback**: Use cleaner synthetic Vietnamese news samples (created via paraphrasing of VnExpress articles) to validate the disentanglement mechanism on a controlled test bed.

### Risk 3 — Single-GPU training time exceeds timeline

**Probability**: Medium (20%).
**Impact**: Medium.
**Fallback**: Reduce to 3 seeds and 4 epochs; report variance honestly. Use the existing post-hoc calibration to reduce training time.

### Risk 4 — Vietnamese↔English MT for back-translation is unavailable or low-quality

**Probability**: Medium-Low (15%).
**Impact**: Medium (loses one RQ).
**Fallback**: Use synonym-swap only (PhoBERT-context-substitute); synonym swap is computationally trivial.

### Risk 5 — Committee asks for source provenance

**Probability**: Very High (95%).
**Impact**: High (if unknown, thesis is weakened).
**Fallback**: Write a thorough Datasheet documenting all observable properties (length, register, attribution, URLs, dates, language, etc.) and acknowledge the gap.

### Risk 6 — The thesis advisor requires ≥2 F1 point improvement on the main test

**Probability**: Medium.
**Impact**: High (the thesis is judged on this).
**Fallback**: Disentanglement may match flat PhoBERT on in-domain test (≤1 F1 drop is acceptable) while improving style-shuffled F1 substantially. Reframe the thesis around robustness, not accuracy.

## L.2 — Risks that are minor

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Test set is too small for paired bootstrap | Low | Low | Bootstrap on 10K iter, ensure sample size ≥ 1000 |
| Stable training diverges after 3 epochs | Medium | Medium | Use early stopping with patience 3 |
| Style auxiliary labels are too noisy | Medium | Low | Use only 2 labels (formal_news / social_media) instead of 3 |
| Reproducibility breaks between machines | Low | Medium | Pin dependencies in requirements.txt with hashes |

---

# Part M — "Do NOT Build Yet"

> Things to **explicitly avoid** implementing before the core research hypothesis (H2) is validated.

### M.1 — Do NOT build retrieval-augmented verification

**Why excluded**: requires Vietnamese trusted corpus construction + Vietnamese NLI, both unavailable in 5–6 months. The thesis fails if retrieval fails.

### M.2 — Do NOT build knowledge-graph reasoning

**Why excluded**: Vietnamese NER quality is the bottleneck. Graph reasoning adds complexity without guaranteed gain.

### M.3 — Do NOT fine-tune Vietnamese LLMs (PhoGPT, VinaLLaMA)

**Why excluded**: zero-shot LLM evaluation is interesting but not architectural. It does not test the disentanglement hypothesis.

### M.4 — Do NOT build an ensemble

**Why excluded**: ensembling is engineering, not research. It hides the contribution.

### M.5 — Do NOT use multilingual models

**Why excluded**: scope is Vietnamese-only. Multilingual models obscure the contribution.

### M.6 — Do NOT add cross-lingual transfer experiments

**Why excluded**: scope is Vietnamese-only.

### M.7 — Do NOT add an active-learning loop

**Why excluded**: dataset is fixed.

### M.8 — Do NOT use GPT-style prompted "explanations" as labels

**Why excluded**: fabricated labels are not defensible.

### M.9 — Do NOT build a temporal-evaluation system

**Why excluded**: only 532 dated samples; insufficient for a thesis-grade temporal study. (Keep temporal split as a small-scale auxiliary analysis, not a chapter.)

### M.10 — Do NOT expand the dataset

**Why excluded**: dataset expansion is research-grade work on its own. The thesis contribution is architectural, not dataset-based.

### M.11 — Do NOT build a web demo before the thesis is validated

**Why excluded**: the existing Streamlit demo is sufficient. Re-deploy after thesis submission if desired.

---

# Part N — Final Supervisor Recommendation

> *If I were the thesis supervisor approving ONE research direction for a 5–6 month Bachelor's thesis based on this exact repository, dataset, and existing implementation, what would I approve?*

---

## 🎯 Decision

### 1. Proposed thesis direction

**Primary**: Direction 2 — Disentangled Style/Content Encoder for robust Vietnamese fake-news detection, with a novel Style-Shuffled Evaluation Protocol.

**Secondary chapter**: Direction 1 — Hierarchical Claim-Aware Encoder.

**Combined final chapter**: Claim-level disentanglement (M4).

### 2. Proposed architecture

```
Article (Vietnamese)
   │
   ├─► Shared PhoBERT encoder (vinai/phobert-base)
   │       outputs: H (B, L, 768)
   │
   ├─► [mean-pool over H, mask-aware]  →  h_pool  (B, 768)
   │
   ├─► Style Head: BiLSTM(128, bidir) → Linear(256, 64)  →  s
   │       auxiliary style classifier on GRL(s) with adversarial loss
   │
   ├─► Content Head: BiLSTM(128, bidir) → Linear(256, 64)  →  c
   │       main classifier: MLP(c) → 2 logits
   │
   └─► Loss: L_CE(MLP(c), y) + λ · L_CE(GRL(s), y_style)
                with λ ramping 0.1 → 1.0 over 4 epochs
```

### 3. Research hypothesis

**H2**: The Disentangled Style/Content Encoder recovers ≥50% of the style-shuffled F1 drop (vs flat PhoBERT), while losing ≤1 F1 point on in-domain test, with p<0.05 across 5 seeds.

### 4. Expected contribution

| Type | Contribution |
|---|---|
| **Architectural** | First adversarial style-content disentanglement applied to Vietnamese fake-news detection |
| **Methodological** | First Style-Shuffled Evaluation Protocol for Vietnamese fake-news detection |
| **Empirical** | First quantification of register-bias in Vietnamese fake-news classifiers |
| **Engineering** | Reusable GradientReversalLayer + style-feature extractor for Vietnamese NLP |
| **Secondary (Direction 1)** | Hierarchical claim-aware encoder with Transformer-Encoder aggregator |

### 5. Required dataset changes

**Minimal**:

- Document deterministic style features for all 13,958 cleaned articles.
- Generate rule-based style auxiliary labels (formal_news / social_media / mixed).
- Build style-shuffled test set (1:1 swap).
- Build paraphrase test set via back-translation (or synonym swap fallback).

**No new annotation required.** No scraping. No dataset expansion.

### 6. Core experiments

| Phase | Experiments |
|---|---|
| **Empirical baseline (RQ1)** | Flat PhoBERT on in-domain test vs style-shuffled test. Report F1, ΔF1. |
| **Disentanglement (RQ2)** | Train Direction 2 with 5 seeds. Compare flat vs disentangled. |
| **Hierarchical (RQ3)** | Train Direction 1 with 5 seeds. Compare flat vs hierarchical. |
| **Combined (RQ4)** | Train M4. Compare flat vs M2 vs M3 vs M4. |
| **Reproducibility (RQ5)** | All runs × 5 seeds. McNemar, Holm-Bonferroni, bootstrap CI, Cohen's d. |
| **Calibration & interpretability (RQ6)** | ECE + claim-level attribution visualisation. |

### 7. Biggest risk

**Disentanglement does not converge or does not improve style-shuffled F1.** If gradient reversal fails on this small dataset (13.9K samples), the central thesis chapter is weakened.

**Mitigation**:

- Pre-validate gradient reversal on a small synthetic Vietnamese dataset before training the full model.
- Keep M1 (PhoBERT + style auxiliary only) as a fallback.
- Treat Direction 1 (Hierarchical) as a parallel contribution that does not depend on disentanglement.

### 8. Fallback plan

If disentanglement fails:

- **Plan A**: Reframe the thesis around **Direction 1 (Hierarchical Claim-Aware)** as the primary contribution. Claim decomposition is independent of disentanglement.
- **Plan B**: Reframe around **empirical register-bias quantification** (RQ1) as the primary contribution, with style-shuffled test set as the methodological artefact.
- **Plan C**: Reframe around **multi-seed reproducibility** (RQ5) as the primary contribution, with the existing 4-model comparison extended to 5 seeds.

Plan A is the most likely fallback if Plan A is partially viable. Plan B is the most defensive.

---

## Honest Verdict

> **This thesis CAN be made research-grade, but only if the student commits to one architectural direction with a clear mechanism, defends it with a novel evaluation protocol, and reports multi-seed results with statistical rigor. The recommended path (Disentangled Style/Content Encoder + Hierarchical Claim-Aware Encoder) is feasible within 5–6 months, addresses a real empirical finding (register-bias), and produces a defensible contribution.**

> **Do NOT** force a more complex architecture (graph, RAG, multilingual) merely because it sounds advanced. The simpler the architecture, the easier it is to defend.

---

*End of Research Architect Report. All findings are grounded in direct inspection of the repository and dataset. Where something is unverifiable, this is stated explicitly.*
