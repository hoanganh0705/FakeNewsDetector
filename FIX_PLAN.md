# Pre-Submission Fix Plan — `fakeNewsDetector`

> Companion document to the pre-submission audit (see chat history).
> This plan turns every **🔴 BLOCKER** and **🟠 HIGH** finding into a concrete,
> testable fix. Items are ordered so that earlier steps unblock later ones
> (you cannot meaningfully run the test suite until `config.py` is restored).
> Nothing in this plan has been executed — read it first, then act.

---

## 0. Ground Rules

Before touching anything:

1. **Verify `.git` is still missing.** The audit ran in a working tree that was *not* a git repo. If a parent `.git` exists elsewhere on disk, prefer recovering the missing files from git history first (it is faster and provably correct):
   ```bash
   # from inside the repo root
   git checkout HEAD -- config.py src/models/   # or whatever the original paths were
   ```
   If git history is unavailable, proceed with the manual restoration below.
2. **Work on a clean branch / copy.** Even though no audit-induced changes remain in the tree (the `fake_news_detector.egg-info/` and `.pytest_cache/` directories created by `pip install` and `pytest` were removed), start any fix work from a fresh snapshot to keep your submission archive reproducible.
3. **One change at a time, then run the verification command listed for that step.** Do not batch.

---

## 1. Restore `config.py`  (🔴 BLOCKER — B1)

### What is wrong
Every Python module imports a top-level `config` package:
```python
from config import cfg           # 27 files including app.py, src/cli.py, src/utils/common.py, ...
```
There is no `config.py` anywhere in the tree. `pyproject.toml` declares:
```toml
[tool.setuptools]
py-modules = ["config"]          # config.py at the root
```
so the file is *meant* to live at the repo root.

### What `config.py` must expose
The audit and the codebase reference these attributes (deduced from every `cfg.X` usage):

| Attribute                | Type      | Notes                                                        |
| ------------------------ | --------- | ------------------------------------------------------------ |
| `cfg.RANDOM_STATE`       | `int`     | Used in `src/utils/common.py.set_reproducibility_seeds`      |
| `cfg.DATA.train_ratio`   | `float`   | Splits must sum to 1.0 (tested in `tests/test_config.py`)    |
| `cfg.DATA.val_ratio`     | `float`   |                                                              |
| `cfg.DATA.test_ratio`    | `float`   |                                                              |
| `cfg.PATHS.raw_data`     | `str`     | `data/raw/raw.csv`                                           |
| `cfg.PATHS.splits_dir`   | `str`     | `data/splits/`                                               |
| `cfg.PATHS.experiments_dir` | `str`  | `experiments/`                                               |
| `cfg.BILSTM.embedding_dim` | `int`   | Positive (tested)                                            |
| `cfg.BILSTM.label_smoothing` | `float`| (tested)                                                     |
| `cfg.BILSTM.fasttext_path`  | `str`  | (tested)                                                     |
| `cfg.PHOBERT.model_name` | `str`     | Non-empty; expected: `vinai/phobert-base`                    |
| `cfg.PHOBERT.label_smoothing` | `float`| (tested)                                                    |
| `cfg`                    | instance of `Config` | Class name `Config`, attribute `Paths` typed as `Paths` (per `tests/test_config.py`) |

### How to reconstruct it

**Option A — recover from git (preferred if available):**
```bash
git checkout HEAD -- config.py
```
If git history does not have it, the file was deleted but never committed, so this will fail — fall through to B.

**Option B — write `config.py` from scratch using the contract above.**

Create `config.py` at the repo root with this minimum skeleton (extend paths/values to match your thesis):

```python
"""
Centralised project configuration.

Single source of truth for random seeds, data paths, model hyperparameters,
and per-model label-smoothing knobs.  All other modules import ``cfg`` from
this module — there must be exactly one instance.

Why top-level (not ``src.config``)?
    ``pyproject.toml`` declares ``py-modules = ["config"]`` so that
    ``pip install -e .`` exposes ``config`` as an importable top-level
    module.  Tests do ``from config import cfg`` directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    raw_data: str = str(PROJECT_ROOT / "data" / "raw" / "raw.csv")
    processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    splits_dir: str = str(PROJECT_ROOT / "data" / "splits")
    features_dir: str = str(PROJECT_ROOT / "data" / "features")
    experiments_dir: str = str(PROJECT_ROOT / "experiments")
    models_dir: str = str(PROJECT_ROOT / "models")
    results_dir: str = str(PROJECT_ROOT / "results")
    figures_dir: str = str(PROJECT_ROOT / "paper" / "figures")
    tables_dir: str = str(PROJECT_ROOT / "paper" / "tables")


# ── data splits ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Data:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = RANDOM_STATE
    min_text_length: int = 5           # used by TextPreprocessor


# ── BiLSTM hyperparameters ────────────────────────────────────────────
@dataclass(frozen=True)
class BiLSTMConfig:
    embedding_dim: int = 300
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    label_smoothing: float = 0.0
    fasttext_path: str = str(PROJECT_ROOT / "data" / "fasttext" / "cc.vi.300.bin")


# ── PhoBERT hyperparameters ───────────────────────────────────────────
@dataclass(frozen=True)
class PhoBERTConfig:
    model_name: str = "vinai/phobert-base"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class Config:
    RANDOM_STATE: int = RANDOM_STATE
    DATA: Data = field(default_factory=Data)
    PATHS: Paths = field(default_factory=Paths)
    BILSTM: BiLSTMConfig = field(default_factory=BiLSTMConfig)
    PHOBERT: PhoBERTConfig = field(default_factory=PhoBERTConfig)


cfg = Config()
```

> Adjust hyperparameter values to whatever you actually used in your experiments — only the **shape** and **types** must match what the tests assert.

### Verify
```bash
python -c "from config import cfg; print(cfg.RANDOM_STATE, cfg.PATHS.raw_data, cfg.BILSTM.embedding_dim, cfg.PHOBERT.model_name)"
python -m pytest tests/test_config.py -v
```
Expected: import prints the values; the `TestConfig` suite passes.

---

## 2. Restore `src/models/` package  (🔴 BLOCKER — B2)

### What is wrong
Five files import model classes from a package that does not exist:

| File                                            | Imports                                               |
| ----------------------------------------------- | ----------------------------------------------------- |
| `src/training/train_bilstm.py`                  | `from src.models.bilstm_model import BiLSTMClassifier` |
| `src/training/train_phobert.py`                 | `from src.models.phobert_model import PhoBertClassifier` |
| `src/training/train_student.py`                 | `from src.models.student_model import StudentBiLSTM`   |
| `src/training/distillation_evaluation.py`       | `BiLSTMClassifier`, `StudentBiLSTM`                   |
| `src/training/reproduce_predictions.py`         | `BiLSTMClassifier`, `PhoBertClassifier`               |
| `app.py` (transitively)                         | imports `src.training.train_bilstm` and `train_phobert`, which in turn need `src.models.*` |

### What each module must expose

The class APIs are determined by how they are *used* in the trainers. Re-read the trainer files (don't change them) and ensure each class has the methods/attributes the trainer touches.

**`src/models/bilstm_model.py` — `BiLSTMClassifier`**
- Constructed with `vocab_size`, `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`, `padding_idx` (or whatever `train_bilstm.py` passes — read it first).
- Standard `nn.Module` with `forward(input_ids, lengths=None) -> logits` (shape `[B, 2]`).
- `save(path)` / `@classmethod load(cls, path, **kwargs)` (used by `BiLSTMTrainer.save/load`).

**`src/models/phobert_model.py` — `PhoBertClassifier`**
- Constructed with `model_name`, `num_labels=2`, `dropout` (whatever `train_phobert.py` passes).
- `forward(input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs)` returning a `transformers.modeling_outputs.SequenceClassifierOutput` (so the trainer can call `loss.backward()` and `logits.argmax(...)`).
- `save_pretrained(path)` / `from_pretrained(path)` (HuggingFace convention).

**`src/models/student_model.py` — `StudentBiLSTM`**
- Same interface as `BiLSTMClassifier` but with a smaller hidden size (used for KD).
- Read `src/training/train_student.py` to confirm exact constructor signature and forward signature.

### How to reconstruct it

**Option A — recover from git (preferred):**
```bash
git checkout HEAD -- src/models/
```
If git history does not have it, this will fail.

**Option B — write the modules from scratch.** Read each trainer file, write a minimal `nn.Module` that satisfies its `forward`, `save`, and `load` calls, and unit-test it before moving on.

### Verify
```bash
python -c "from src.models.bilstm_model import BiLSTMClassifier; \
           from src.models.phobert_model import PhoBertClassifier; \
           from src.models.student_model import StudentBiLSTM; \
           print('OK')"
python -c "from src.training import train_bilstm, train_phobert, train_student, distillation_evaluation, reproduce_predictions; print('imports OK')"
```
Expected: both print OK with no `ModuleNotFoundError`.

---

## 3. Fix the `fasttext-wheel` install failure  (🔴 BLOCKER — B3)

### What is wrong
`pip install -e .` fails because `fasttext-wheel==0.9.2` cannot compile its C++ extension on Python ≥ 3.13 (verified end-to-end with Python 3.14.7 — `src/args.cc:474` C++ compile error).

### Decision matrix — pick one

| Option | When to pick it | Effort | Risk |
| ------ | --------------- | ------ | ---- |
| **A. Pin Python `<3.13`** | You want minimal changes; reviewers can use Python 3.10/3.11/3.12 | Trivial | Reviewer on Python 3.13+ cannot install |
| **B. Drop `fasttext` entirely** | BiLSTM uses random-init embeddings (the codebase already supports this) | Small | Loses FastText-init result in thesis (rerun ablation) |
| **C. Use source-built `fasttext` from a working commit** | You need FastText embeddings and want Python 3.13+ support | Medium | Must verify behaviour matches 0.9.2 |

### Recommended: Option A (lowest risk, fastest)

Edit **`pyproject.toml`**:
```toml
requires-python = ">=3.10,<3.13"
```
Edit **`README.md`** Quick Start to call out the supported range:
```diff
- pip install -r requirements.txt
- export JAVA_HOME=/usr/lib/jvm/java-25-openjdk  # adjust to your Java path
+ # Requires Python 3.10, 3.11, or 3.12.  Python 3.13+ is not yet supported
+ # because fasttext-wheel 0.9.2 does not build against its C++ headers.
+ pip install -r requirements.txt
+ export JAVA_HOME=/usr/lib/jvm/java-25-openjdk  # adjust to your Java path
```

### Verify
```bash
# In a fresh virtualenv with Python 3.11:
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "import fasttext; print('fasttext OK')"
```
Expected: install completes with no errors; `fasttext` imports cleanly.

---

## 4. Fix the README↔repository mismatch  (🟠 HIGH — H2, H3, L3)

### H2 — `MANUAL.md` does not exist
**Fix A (preferred):** create `MANUAL.md` containing the step-by-step reproduction workflow that the README promises. Minimum content:
1. `pip install -e .` (Python 3.10–3.12)
2. `fakenews preprocess`        # word segmentation (downloads VnCoreNLP jar)
3. `fakenews split`             # 70/15/15 stratified split
4. `fakenews features`          # TF-IDF, embeddings, PhoBERT tokenisation
5. `fakenews train all`         # LR, SVM, BiLSTM, PhoBERT
6. `fakenews evaluate`          # metrics + figures
7. `fakenews calibration`       # ECE, MCE, Brier
8. `fakenews explain`           # SHAP / IG / attention rollout (Phase 1)
9. `fakenews recalibrate`       # post-hoc calibration (Phase 2)
10. `fakenews distill --mode all`  # KD student (Phase 3)
11. `fakenews hard-cases`        # error deep-dive (Phase 4)
12. `streamlit run app.py`       # demo

**Fix B (fallback if you don't want a separate manual):** remove the reference from `README.md`:
```diff
-See [MANUAL.md](MANUAL.md) for the full step-by-step guide.
+See the **Quick Start** below for the step-by-step guide.
```
…and inline the steps above into README's Quick Start.

### H3 — Project-structure tree promises files that are gitignored
Edit the Project Structure section in `README.md` to make it honest:

```diff
 ├── data/
-│   ├── raw/raw.csv                     # Original dataset
-│   ├── processed/segmented.csv         # Word-segmented text
-│   ├── splits/                         # Train/Val/Test splits (70/15/15)
-│   └── features/                       # Extracted features
+│   └── raw/raw.csv                     # Original dataset (15,789 rows)
+│
+# Generated by `fakenews preprocess` / `fakenews split` / `fakenews features`:
+#   data/processed/segmented.csv
+#   data/splits/{train,val,test}.csv
+#   data/features/{tfidf,embedding,phobert}/
```

### L3 — `paper/` directory missing from working tree
Two options:

**Option 1 (preferred for completeness):** include the `paper/` directory in the submission archive. It contains LaTeX source, figures, and tables that back up the thesis.

**Option 2:** scope README's `paper/` mentions to the *compiled PDF only* (which is already in the project's tracked status), or note that the LaTeX source is available upon request.

### Verify
```bash
# Confirm every relative link in README resolves:
grep -oE '\[[^]]+\]\([^)]+\)' README.md | grep -oE '\([^)]+\)' | sort -u | while read link; do
  target="${link#(}"; target="${target%)}"
  case "$target" in http*|mailto:*|\#*) continue;; esac
  [ -e "$target" ] || echo "MISSING: $target"
done
```
Expected: no `MISSING:` lines.

---

## 5. Restore or scope-down `.git/`  (🟠 HIGH — H1)

### What is wrong
The working tree has **no `.git/` directory** at all. This means:
- We cannot tell what is tracked vs. untracked.
- The `D config.py` / missing `src/models/` were almost certainly *deleted from a tracked state*.

### Recommended action

1. **Before submitting**, attempt to restore the repository to a known-good state from version control:
   ```bash
   git clone <your-origin-url> /tmp/fnd-restore
   rsync -a /tmp/fnd-restore/ ./   # over the current working tree
   ```
   Verify with `git status` — there should be no `D` lines for `config.py` and `src/models/`.

2. **For the submission archive** (zip/tar you hand to the instructor), exclude `.git/`:
   ```bash
   tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
       --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' \
       --exclude='.venv' --exclude='*.egg-info' \
       --exclude='paper/figures/*.pdf' \
       -czf FakeNewsDetector-submission.tar.gz FakeNewsDetector/
   ```

### Verify
```bash
tar -tzf FakeNewsDetector-submission.tar.gz | grep -E "\.git/" || echo "OK: no .git/ in archive"
[ -d .git ] && git status | grep -E "^[ ]*D.*config.py|^[ ]*D.*src/models" && echo "STILL MISSING" || echo "OK: files restored"
```

---

## 6. Strengthen `.gitignore`  (🟡 MEDIUM — M3)

### What is wrong
The audit's own `pytest` invocation created `.pytest_cache/`. The repo's `.gitignore` covers `__pycache__/`, `*.py[cod]`, `.venv/`, IDE files, model checkpoints, logs, generated data, experiments, results — but **not** the standard Python-tooling caches.

### Fix
Append to **`.gitignore`**:
```gitignore
# Test caches
.pytest_cache/
.coverage
htmlcov/

# Linter / type-checker caches
.ruff_cache/
.mypy_cache/

# Build outputs
dist/
build/
```

### Verify
```bash
python -m pytest tests/test_config.py    # creates .pytest_cache/
git status --short                       # should NOT list .pytest_cache/
```

---

## 7. Documentation nits  (🔵 LOW — L1, L2)

These are stylistic and only matter if you have spare time:

* **L1 — `paper/` in archive**: already covered by §5 above.
* **L2 — `app.py` README claim**: after restoring `src/models/` (B2) the demo will import successfully; no further README change is strictly needed. The README already correctly states "If a checkpoint or feature file is missing, the demo shows a clear error explaining which artefact is required."

---

## 8. Final Acceptance Gate

After completing §§ 1–6, run this end-to-end gate before declaring the repository ready:

```bash
# 1. Clean install in a fresh virtualenv on Python 3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Module import sanity
python -c "from config import cfg; print(cfg.RANDOM_STATE)"
python -c "from src.models.bilstm_model import BiLSTMClassifier; \
           from src.models.phobert_model import PhoBertClassifier; \
           from src.models.student_model import StudentBiLSTM"

# 3. Full test suite
python -m pytest -v

# 4. CLI entry point
fakenews --help

# 5. Dataset sanity
python -c "import pandas as pd; print(pd.read_csv('data/raw/raw.csv').shape)"

# 6. Smoke-run the CLI on a tiny sample (optional but recommended)
fakenews preprocess
fakenews split
fakenews features
fakenews train lr
fakenews evaluate
```

A clean run of all six steps is the bar for **🟢 READY TO SUBMIT**.

---

## 9. Effort Estimate

| Step                                                | Skill level    | Time          |
| --------------------------------------------------- | -------------- | ------------- |
| §1. Restore `config.py`                             | Intermediate   | 30–60 min     |
| §2. Restore `src/models/`                           | Intermediate+  | 1–3 hours     |
| §3. Pin Python `<3.13` in `pyproject.toml`          | Trivial        | 5 min         |
| §4. Fix README ↔ repo mismatches                    | Easy           | 30 min        |
| §5. Restore / scope `.git/`                         | Easy           | 15 min        |
| §6. Strengthen `.gitignore`                         | Trivial        | 5 min         |
| §8. Acceptance gate                                 | Easy           | 30 min        |
| **Total**                                           |                | **~4–6 hours** |

If git history is intact, §§ 1 and 2 drop to *seconds* (a single `git checkout`) and the total drops to **~1.5 hours**.

---

## 10. Final Classification After Fixes

If §§ 1–6 are completed:

| Severity | Items remaining after fix |
| -------- | ------------------------- |
| 🔴 BLOCKER | 0 (all three addressed) |
| 🟠 HIGH    | 0 (H1, H2, H3 all addressed) |
| 🟡 MEDIUM  | 0 (M1–M3 are by-products of B1/B2; M3 covered by §6) |
| 🔵 LOW     | 0 (L1 covered by §5; L2 covered by §2) |
| ❓ UNKNOWN | U1 (dataset licence), U2 (VnCoreNLP jar download doc), U3 (PhoBERT download doc) — **these should be addressed by adding one or two sentences to `MANUAL.md` / `README.md` during §4** |

**Expected verdict after fixes:** 🟢 **READY TO SUBMIT** (assuming §§ 1–6 pass the acceptance gate in §8).

---

## 11. Phased Implementation Plan

The §1–§10 fixes are organised below into **four phases**. Phases must be done in order — each phase depends on the previous one being green. Every phase ends with a **Phase Exit Gate** that you must pass before starting the next.

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0 — Recovery & Sanity           (~5 min)  ← cheapest, do first  │
│ Phase 1 — Restore Missing Source      (~30 min – 3 hrs)                │
│ Phase 2 — Fix Install & Packaging     (~15 min)                        │
│ Phase 3 — Documentation & Hygiene     (~45 min)                        │
│ Phase 4 — Final Acceptance Gate       (~30 min)  ← gates submission    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 0 — Recovery & Sanity  (~5 min)

**Goal:** determine whether git history can do the heavy lifting for us before any manual work begins.

| Step | Action | Command |
| ---- | ------ | ------- |
| 0.1  | Check if a git remote exists or a backup is available. | `git remote -v` |
| 0.2  | If yes, clone to a scratch directory. | `git clone <url> /tmp/fnd-restore` |
| 0.3  | Verify `config.py` and `src/models/` exist there. | `ls /tmp/fnd-restore/config.py /tmp/fnd-restore/src/models/` |
| 0.4  | If they exist, restore them in-place. | `cp /tmp/fnd-restore/config.py ./config.py` and `cp -r /tmp/fnd-restore/src/models ./src/models` |
| 0.5  | If no git history, proceed to Phase 1 with the manual reconstruction route.**

**Phase 0 Exit Gate:**
- ✅ At least one of these is true: `config.py` now exists at the repo root, **or** you have a confirmed plan to reconstruct it from scratch.
- ✅ `src/models/` either exists or has a confirmed reconstruction plan.

> **If Phase 0 short-circuits** (git history had everything), skip the manual parts of Phase 1 and go straight to Phase 2. Total remaining work drops to ~1 hour.

---

### Phase 1 — Restore Missing Source  (~30 min – 3 hrs)

**Goal:** bring the codebase to a state where every `import` resolves, even before any test runs.

#### 1A. Restore `config.py` (B1)

| Sub-step | Action |
| -------- | ------ |
| 1A.1 | If you skipped Phase 0, follow §1 of this document. Create `config.py` at the repo root matching the `cfg.*` attribute contract. |
| 1A.2 | Run the import sanity check. |
| 1A.3 | Run `tests/test_config.py` — should pass all `TestConfig` cases (random state is int, splits sum to 1, paths are strings, etc.). |

**Verify:**
```bash
python -c "from config import cfg; print(cfg.RANDOM_STATE, cfg.PATHS.raw_data)"
python -m pytest tests/test_config.py -v
```

#### 1B. Restore `src/models/` (B2)

| Sub-step | Action |
| -------- | ------ |
| 1B.1 | Re-read each affected trainer file (`train_bilstm.py`, `train_phobert.py`, `train_student.py`, `distillation_evaluation.py`, `reproduce_predictions.py`) and list every method/attribute the trainers use on the model class. |
| 1B.2 | Write or restore `src/models/bilstm_model.py` with `BiLSTMClassifier`, `src/models/phobert_model.py` with `PhoBertClassifier`, `src/models/student_model.py` with `StudentBiLSTM`. |
| 1B.3 | Add `src/models/__init__.py` (can be empty or re-export the three classes). |
| 1B.4 | Run the import sanity check for trainers + evaluation scripts. |

**Verify:**
```bash
python -c "from src.models.bilstm_model import BiLSTMClassifier; \
           from src.models.phobert_model import PhoBertClassifier; \
           from src.models.student_model import StudentBiLSTM"
python -c "from src.training import train_bilstm, train_phobert, train_student, distillation_evaluation, reproduce_predictions"
```

**Phase 1 Exit Gate:**
- ✅ `from config import cfg` succeeds.
- ✅ `from src.models.* import ...` succeeds for all three classes.
- ✅ All five trainer/evaluation modules import without `ModuleNotFoundError`.
- ✅ `tests/test_config.py` passes.

---

### Phase 2 — Fix Install & Packaging  (~15 min)

**Goal:** `pip install -e .` succeeds in a clean virtualenv on a supported Python version.

| Sub-step | Action |
| -------- | ------ |
| 2.1 | Edit `pyproject.toml` — change `requires-python = ">=3.10"` to `requires-python = ">=3.10,<3.13"` (the recommended Option A from §3). |
| 2.2 | Mirror the same constraint in `README.md` Quick Start. |
| 2.3 | Create a fresh virtualenv on Python 3.11. |
| 2.4 | Install the project and run the sanity checks. |

**Verify (from a fresh shell):**
```bash
python3.11 -m venv .venv-test && source .venv-test/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
python -c "import fasttext, torch, transformers, underthesea, py_vncorenlp; print('all deps OK')"
python -c "from config import cfg; print(cfg.RANDOM_STATE)"
python -m pytest tests/test_config.py -v
deactivate && rm -rf .venv-test
```

**Phase 2 Exit Gate:**
- ✅ `pip install -e .` completes with exit code 0.
- ✅ All heavy dependencies (fasttext, torch, transformers, underthesea, py_vncorenlp) import cleanly.
- ✅ Tests in `tests/test_config.py` still pass.

---

### Phase 3 — Documentation & Hygiene  (~45 min)

**Goal:** the submission is internally consistent — README matches reality, no junk files leak into the archive.

#### 3A. README reconciliation (H2, H3, L3)

| Sub-step | Action |
| -------- | ------ |
| 3A.1 | Either write `MANUAL.md` (Fix A from §4) **or** remove the `MANUAL.md` link from `README.md` (Fix B). |
| 3A.2 | Update the Project Structure tree to show only `data/raw/raw.csv` as shipped, and note that `data/processed/`, `data/splits/`, `data/features/` are generated. |
| 3A.3 | Add one-line notes about VnCoreNLP jar download (U2) and PhoBERT model download (U3). |
| 3A.4 | Document the dataset source/licence if known, or mark it `UNKNOWN — see thesis for source attribution` (U1). |
| 3A.5 | Run the README-link validator from §4. |

#### 3B. `.gitignore` hardening (M3)

| Sub-step | Action |
| -------- | ------ |
| 3B.1 | Append `.pytest_cache/`, `.coverage`, `htmlcov/`, `.ruff_cache/`, `.mypy_cache/`, `dist/`, `build/` to `.gitignore`. |
| 3B.2 | Verify by running pytest and confirming `git status --short` (if git is present) does not list `.pytest_cache/`. |

#### 3C. Archive preparation (L1, §5)

| Sub-step | Action |
| -------- | ------ |
| 3C.1 | Ensure `.git/` is included in the repo (so reviewers can `git log`), but excluded from the **submission archive**. |
| 3C.2 | Produce the submission archive with the canonical tar recipe from §5. |
| 3C.3 | Verify: `tar -tzf ... \| grep .git/` returns nothing. |

**Phase 3 Exit Gate:**
- ✅ Every markdown link in `README.md` resolves.
- ✅ No junk files (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`) are tracked or present in the archive.
- ✅ Dataset source/licence is either documented or explicitly marked `UNKNOWN`.

---

### Phase 4 — Final Acceptance Gate  (~30 min)

**Goal:** a single end-to-end run that proves the submission is ready.

This is the gate from §8, reproduced here as the formal Phase 4 plan. **All six steps must pass on the first attempt.**

| Step | Command | Pass criterion |
| ---- | ------- | -------------- |
| 4.1 | `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e .` | Exit code 0; no errors. |
| 4.2 | `python -c "from config import cfg; print(cfg.RANDOM_STATE, cfg.PATHS.raw_data)"` | Prints expected values. |
| 4.3 | `python -c "from src.models.bilstm_model import BiLSTMClassifier; from src.models.phobert_model import PhoBertClassifier; from src.models.student_model import StudentBiLSTM"` | No `ModuleNotFoundError`. |
| 4.4 | `python -m pytest -v` | All tests pass (or only doc-style/skip). |
| 4.5 | `fakenews --help` | Prints the CLI help. |
| 4.6 | `fakenews preprocess && fakenews split && fakenews features && fakenews train lr && fakenews evaluate` | Pipeline completes end-to-end on at least LR (cheapest model). |
| 4.7 | Archive check: `tar -tzf FakeNewsDetector-submission.tar.gz \| grep .git/` | Empty output. |
| 4.8 | `tar -tzf FakeNewsDetector-submission.tar.gz \| grep -E "config\.py\|src/models/"` | Both present. |

**Phase 4 Exit Gate:**
- ✅ Every numbered step in the table above passes.
- ✅ If any step fails, **stop**. Diagnose, fix, re-run from step 4.1. Do not proceed to submission.

---

### Phase Dependency Graph

```
        ┌─────────────┐
        │  Phase 0    │ Recovery & sanity
        └──────┬──────┘
               │ exit gate: config.py + src/models/ exist or have plan
               ▼
        ┌─────────────┐
        │  Phase 1    │ Restore missing source (config.py, src/models)
        └──────┬──────┘
               │ exit gate: all imports + tests/test_config.py pass
               ▼
        ┌─────────────┐
        │  Phase 2    │ Fix install (pin Python, fresh venv)
        └──────┬──────┘
               │ exit gate: pip install -e . succeeds
               ▼
        ┌─────────────┐
        │  Phase 3    │ Docs & hygiene (README, gitignore, archive)
        └──────┬──────┘
               │ exit gate: README links resolve, archive clean
               ▼
        ┌─────────────┐
        │  Phase 4    │ Final acceptance gate
        └──────┬──────┘
               │ exit gate: all 8 steps pass
               ▼
           🟢 READY TO SUBMIT
```

### Phase 0 + 1 Combined Effort (best / worst case)

| Scenario | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Total** |
| -------- | ------- | ------- | ------- | ------- | ------- | --------- |
| Git history intact (best case) | 5 min | 0 min | 15 min | 45 min | 30 min | **~1.5 hrs** |
| No git history, full manual (worst) | 5 min | ~3 hrs | 15 min | 45 min | 30 min | **~5 hrs** |

### Hand-off Checklist

When you finish Phase 4, attach a one-page summary to the submission containing:

1. ✅ Confirmation that all 4 phases passed their exit gates (with timestamps).
3. ✅ The final tarball filename and SHA256:
   ```bash
   sha256sum FakeNewsDetector-submission.tar.gz
   ```
4. ✅ The output of `fakenews --help` (one screenshot / paste) as proof the entry point works.
5. ✅ The output of `python -m pytest -v` (last 20 lines) as proof the test suite runs.

This is what you hand to the instructor alongside the tarball.
