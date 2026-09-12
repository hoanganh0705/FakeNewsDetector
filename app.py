"""
Streamlit demo for the Vietnamese Fake News Detector research project.

This module is an inference/demo layer ONLY — it loads the *existing*
trained model artifacts produced by ``fakenews train …`` and the existing
preprocessing / feature-extraction modules.  No model is retrained and no
existing artefact is modified.

Usage::

    source .venv/bin/activate
    streamlit run app.py

Label convention (preserved from the training pipeline):
    0 = Real
    1 = Fake
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import streamlit as st

# Ensure the project root is on sys.path when launched via ``streamlit run``.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import cfg  # noqa: E402


# ============================================================================
# Constants
# ============================================================================

LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "Real", LABEL_FAKE: "Fake"}
LABEL_COLORS = {LABEL_REAL: "#1f8a4c", LABEL_FAKE: "#c0392b"}

# Static research benchmark (from THESIS_PROJECT_CONTEXT.md §7).
RESEARCH_METRICS = {
    "PhoBERT":            {"accuracy": 0.9007, "f1": 0.8988, "auc": 0.9495, "best": True},
    "SVM":                {"accuracy": 0.8434, "f1": 0.8410, "auc": 0.9190},
    "Logistic Regression":{"accuracy": 0.8348, "f1": 0.8329, "auc": 0.9188},
    "BiLSTM":             {"accuracy": 0.8252, "f1": 0.8232, "auc": 0.9048},
}
BEST_MODEL_NAME = "PhoBERT"

MODEL_KEYS = ("lr", "svm", "bilstm", "phobert")
DISPLAY = {"lr": "Logistic Regression", "svm": "SVM",
           "bilstm": "BiLSTM", "phobert": "PhoBERT"}


# ============================================================================
# Custom exception — surfaced as a friendly UI error
# ============================================================================

class ArtifactMissingError(RuntimeError):
    """Raised when a model/feature artefact required for inference is absent."""


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class Prediction:
    """Single-model prediction result."""
    label: int                 # 0 = Real, 1 = Fake
    p_real: float              # P(Real)
    p_fake: float              # P(Fake)
    confidence: float          # max(p_real, p_fake)
    n_tokens: Optional[int] = None  # tokenizer / vocab token count if available

    @property
    def name(self) -> str:
        return LABEL_NAMES[self.label]

    def as_row(self, model_name: str) -> dict:
        return {
            "Model":      model_name,
            "Prediction": self.name,
            "Confidence": f"{self.confidence * 100:.2f}%",
            "P(Real)":    f"{self.p_real * 100:.2f}%",
            "P(Fake)":    f"{self.p_fake * 100:.2f}%",
        }


# ============================================================================
# Pre-processing
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_text_preprocessor():
    """Lazy-load (and cache) the project's text cleaner."""
    from src.preprocessing.text_preprocessor import TextPreprocessor
    return TextPreprocessor()


def preprocess_text(raw_text: str) -> str:
    """Apply the project's text-cleaning pipeline.

    Word-segmentation is intentionally NOT applied here — the TF-IDF and
    BiLSTM training pipelines segment at train time, but for a single
    inference call segmentation latency is non-trivial and the LR/SVM
    vectorizer was fitted on segmented text while BiLSTM uses the same
    vocabulary.  The cleaner used here matches the first stage of
    preprocessing used at training time.
    """
    if not raw_text or not raw_text.strip():
        return ""
    pre = get_text_preprocessor()
    return pre.clean_text(raw_text)


# ============================================================================
# Model loaders (lazy, cached)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_lr():
    """Load the Logistic Regression model + its fitted TF-IDF vectorizer."""
    model_path = os.path.join(cfg.PATHS.lr_dir, "lr_model.pkl")
    vec_path = os.path.join(cfg.PATHS.tfidf_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path):
        raise ArtifactMissingError(
            f"LR model checkpoint not found: {model_path}\n"
            "Train it first with:  fakenews train lr"
        )
    if not os.path.exists(vec_path):
        raise ArtifactMissingError(
            f"TF-IDF vectorizer not found: {vec_path}\n"
            "Run feature extraction first:  fakenews features"
        )
    import joblib
    blob = joblib.load(model_path)
    from src.features.tfidf_features import TfidfFeatureExtractor
    extractor = TfidfFeatureExtractor.load(vec_path)
    return {"model": blob["model"], "vectorizer": extractor}


@st.cache_resource(show_spinner=False)
def load_svm():
    """Load the SVM model + its fitted TF-IDF vectorizer."""
    model_path = os.path.join(cfg.PATHS.svm_dir, "svm_model.pkl")
    vec_path = os.path.join(cfg.PATHS.tfidf_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path):
        raise ArtifactMissingError(
            f"SVM model checkpoint not found: {model_path}\n"
            "Train it first with:  fakenews train svm"
        )
    if not os.path.exists(vec_path):
        raise ArtifactMissingError(
            f"TF-IDF vectorizer not found: {vec_path}\n"
            "Run feature extraction first:  fakenews features"
        )
    import joblib
    blob = joblib.load(model_path)
    from src.features.tfidf_features import TfidfFeatureExtractor
    extractor = TfidfFeatureExtractor.load(vec_path)
    return {"model": blob["model"], "vectorizer": extractor}


@st.cache_resource(show_spinner=False)
def load_bilstm():
    """Load the BiLSTM trainer (model + vocabulary + FastText)."""
    import torch
    model_path = os.path.join(cfg.PATHS.bilstm_dir, "bilstm_model.pt")
    extractor_path = os.path.join(cfg.PATHS.embedding_dir, "embedding_extractor.pkl")
    if not os.path.exists(model_path):
        raise ArtifactMissingError(
            f"BiLSTM checkpoint not found: {model_path}\n"
            "Train it first with:  fakenews train bilstm"
        )
    if not os.path.exists(extractor_path):
        raise ArtifactMissingError(
            f"Embedding extractor not found: {extractor_path}\n"
            "Run feature extraction first:  fakenews features"
        )

    from src.training.train_bilstm import BiLSTMTrainer
    from src.features.embedding_features import EmbeddingFeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = BiLSTMTrainer.load(model_path, device=str(device))
    trainer.model.to(device)
    trainer.model.eval()
    extractor = EmbeddingFeatureExtractor.load(extractor_path)
    return {"trainer": trainer, "extractor": extractor, "device": device}


@st.cache_resource(show_spinner=False)
def load_phobert():
    """Load the PhoBERT trainer + tokenizer (no raw text I/O needed)."""
    import torch
    model_path = os.path.join(cfg.PATHS.bert_dir, "phobert_model.pt")
    if not os.path.exists(model_path):
        raise ArtifactMissingError(
            f"PhoBERT checkpoint not found: {model_path}\n"
            "Train it first with:  fakenews train phobert"
        )

    from src.training.train_phobert import PhoBertTrainer
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PhoBertTrainer.load(model_path, device=str(device))
    trainer.model.to(device)
    trainer.model.eval()

    # Tokenizer: prefer local cache (set up by feature extraction), else Hub.
    local_cache = os.path.join(cfg.PATHS.features_dir, "phobert_tokenizer_cache")
    try:
        if os.path.isdir(local_cache) and os.listdir(local_cache):
            tokenizer = AutoTokenizer.from_pretrained(local_cache)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.PHOBERT.model_name)
    except Exception as exc:  # pragma: no cover - depends on connectivity
        raise ArtifactMissingError(
            f"Could not load PhoBERT tokenizer ({cfg.PHOBERT.model_name}): {exc}"
        ) from exc

    return {"trainer": trainer, "tokenizer": tokenizer, "device": device}


# ============================================================================
# Inference helpers
# ============================================================================

def _safe_predict_proba_pfake(model, X) -> float:
    """Return P(Fake) regardless of which sklearn class-index that is."""
    proba = model.predict_proba(X)
    classes_ = list(getattr(model, "classes_", [0, 1]))
    try:
        idx_fake = classes_.index(LABEL_FAKE)
    except ValueError:
        idx_fake = 1  # fall back to last column
    return float(np.asarray(proba)[0, idx_fake])


def predict_lr(clean_text: str) -> Prediction:
    bundle = load_lr()
    X = bundle["vectorizer"].transform([clean_text])
    p_fake = _safe_predict_proba_pfake(bundle["model"], X)
    return Prediction(
        label=LABEL_FAKE if p_fake >= 0.5 else LABEL_REAL,
        p_real=1.0 - p_fake,
        p_fake=p_fake,
        confidence=max(p_fake, 1.0 - p_fake),
    )


def predict_svm(clean_text: str) -> Prediction:
    bundle = load_svm()
    X = bundle["vectorizer"].transform([clean_text])
    p_fake = _safe_predict_proba_pfake(bundle["model"], X)
    return Prediction(
        label=LABEL_FAKE if p_fake >= 0.5 else LABEL_REAL,
        p_real=1.0 - p_fake,
        p_fake=p_fake,
        confidence=max(p_fake, 1.0 - p_fake),
    )


def predict_bilstm(clean_text: str) -> Prediction:
    import torch
    bundle = load_bilstm()
    trainer, extractor, device = bundle["trainer"], bundle["extractor"], bundle["device"]

    indices = extractor.vocab.text_to_indices(clean_text)
    if not indices:
        # Nothing to score → return neutral prediction
        return Prediction(label=LABEL_REAL, p_real=1.0, p_fake=0.0,
                          confidence=1.0, n_tokens=0)

    max_len = extractor.max_seq_length
    if len(indices) > max_len:
        indices = indices[:max_len]

    seq = torch.tensor([indices], dtype=torch.long, device=device)
    mask = (seq != 0).long()
    with torch.inference_mode():
        logits = trainer.model(seq, mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    p_real, p_fake = float(probs[0]), float(probs[1])
    label = LABEL_FAKE if p_fake >= p_real else LABEL_REAL
    return Prediction(label=label, p_real=p_real, p_fake=p_fake,
                      confidence=max(p_real, p_fake), n_tokens=len(indices))


def predict_phobert(clean_text: str) -> Prediction:
    import torch
    bundle = load_phobert()
    trainer, tokenizer, device = bundle["trainer"], bundle["tokenizer"], bundle["device"]

    enc = tokenizer(
        [clean_text],
        padding="max_length",
        truncation=True,
        max_length=int(cfg.PHOBERT.max_seq_len),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.inference_mode():
        logits = trainer.model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    p_real, p_fake = float(probs[0]), float(probs[1])
    label = LABEL_FAKE if p_fake >= p_real else LABEL_REAL
    n_tokens = int(attention_mask.sum().item())
    return Prediction(label=label, p_real=p_real, p_fake=p_fake,
                      confidence=max(p_real, p_fake), n_tokens=n_tokens)


PREDICTORS = {
    "lr":      predict_lr,
    "svm":     predict_svm,
    "bilstm":  predict_bilstm,
    "phobert": predict_phobert,
}

LOADERS = {
    "lr":      load_lr,
    "svm":     load_svm,
    "bilstm":  load_bilstm,
    "phobert": load_phobert,
}


# ============================================================================
# Text statistics
# ============================================================================

def text_stats(raw_text: str, token_count: Optional[int] = None) -> dict:
    cleaned = preprocess_text(raw_text) if raw_text else ""
    return {
        "characters":  len(raw_text or ""),
        "words":       len((raw_text or "").split()),
        "cleaned_chars": len(cleaned),
        "tokens":      token_count,
    }


# ============================================================================
# Device information
# ============================================================================

def device_info() -> dict:
    """Return human-readable device information for the sidebar."""
    import torch
    info = {"cuda": bool(torch.cuda.is_available()), "device": "CPU"}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["device"]  = torch.cuda.get_device_name(idx)
        info["count"]   = torch.cuda.device_count()
        info["index"]   = idx
    return info


# ============================================================================
# UI rendering helpers
# ============================================================================

def render_prediction(pred: Prediction) -> None:
    """Render the main REAL/FAKE prediction card."""
    label_color = LABEL_COLORS[pred.label]
    label_name = LABEL_NAMES[pred.label]

    st.markdown(
        f"""
        <div style="
            padding: 24px;
            border-radius: 12px;
            background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
            border-left: 8px solid {label_color};
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 16px;
        ">
            <div style="font-size: 14px; color: #6b7280; letter-spacing: 0.05em;
                        text-transform: uppercase; font-weight: 600;">
                Prediction
            </div>
            <div style="font-size: 56px; font-weight: 800; color: {label_color};
                        line-height: 1.1; margin-top: 4px;">
                {label_name}
            </div>
            <div style="font-size: 16px; color: #374151; margin-top: 8px;">
                Confidence: <strong>{pred.confidence * 100:.2f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability visualisation
    st.markdown("**Probability distribution**")
    col_r, col_f = st.columns(2)
    col_r.progress(min(max(pred.p_real, 0.0), 1.0), text=f"P(Real) {pred.p_real * 100:.2f}%")
    col_f.progress(min(max(pred.p_fake, 0.0), 1.0), text=f"P(Fake) {pred.p_fake * 100:.2f}%")


def render_comparison(results: dict) -> None:
    """Render the side-by-side comparison table for All Models mode."""
    import pandas as pd

    rows = [pred.as_row(DISPLAY[k]) for k, pred in results.items()]
    df = pd.DataFrame(rows)

    st.markdown("### Model comparison")
    st.dataframe(df, use_container_width=True, hide_index=True)

    fake_votes = sum(1 for p in results.values() if p.label == LABEL_FAKE)
    real_votes = sum(1 for p in results.values() if p.label == LABEL_REAL)
    n = len(results)

    agreement_color = "#1f8a4c" if fake_votes in (0, n) or real_votes in (0, n) else "#b7791f"
    if fake_votes == n:
        msg, color = f"{n} / {n} models predict FAKE", LABEL_COLORS[LABEL_FAKE]
    elif real_votes == n:
        msg, color = f"{n} / {n} models predict REAL", LABEL_COLORS[LABEL_REAL]
    elif fake_votes > real_votes:
        msg, color = f"{fake_votes} / {n} models predict FAKE", LABEL_COLORS[LABEL_FAKE]
    else:
        msg, color = f"{real_votes} / {n} models predict REAL", LABEL_COLORS[LABEL_REAL]
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{color}15;"
        f"border-left:6px solid {color};font-weight:600;color:{color};'>"
        f"Model agreement &mdash; {msg}</div>",
        unsafe_allow_html=True,
    )

    # Highlight the best model (PhoBERT) when available
    if "phobert" in results:
        phobert_pred = results["phobert"]
        st.markdown("---")
        st.markdown(f"### Highlight — {BEST_MODEL_NAME} (best model)")
        st.caption(
            f"{BEST_MODEL_NAME} achieves the highest benchmark F1 (0.8988) "
            f"and accuracy (90.07%) on the test set."
        )
        render_prediction(phobert_pred)


def render_text_stats(stats: dict) -> None:
    st.markdown("### Text statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters", stats["characters"])
    c2.metric("Words", stats["words"])
    c3.metric("Cleaned chars", stats["cleaned_chars"])
    c4.metric("Tokens", "—" if stats["tokens"] is None else stats["tokens"])


def render_research_benchmark() -> None:
    import pandas as pd
    rows = []
    for name, m in RESEARCH_METRICS.items():
        rows.append({
            "Model":    f"⭐ {name}" if m.get("best") else name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "F1":       f"{m['f1'] * 100:.2f}%",
            "AUC":      f"{m['auc']:.4f}",
        })
    st.markdown("### Research benchmark (test set)")
    st.caption(
        "These are the published benchmark scores from the existing "
        "research experiments. They are **not** recomputed from the current "
        "user input."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar() -> str:
    """Render the sidebar and return the selected model key."""
    with st.sidebar:
        st.markdown("## 🔬 FakeNewsDetector")
        st.caption("Vietnamese fake-news research demo")

        st.markdown("---")
        st.markdown("### Model")
        choice = st.radio(
            "Choose a model",
            options=["all", "lr", "svm", "bilstm", "phobert"],
            format_func=lambda k: "All Models" if k == "all" else DISPLAY[k],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### System")
        info = device_info()
        st.write(f"**Device:** {info['device']}")
        st.write(f"**CUDA available:** {'Yes' if info['cuda'] else 'No'}")
        if info["cuda"]:
            st.write(f"**GPU index:** {info['index']}")

        st.markdown("---")
        with st.expander("About this project", expanded=False):
            st.markdown(
                "**Vietnamese Fake News Detection**\n\n"
                "Comparative study of ML, deep-learning and transformer "
                "models on a 15K-article Vietnamese corpus.\n\n"
                "*Best model:* PhoBERT (90.07% accuracy, F1 0.8988).\n\n"
                "This Streamlit demo runs the **existing** trained models "
                "without any retraining."
            )

        with st.expander("Research metrics (test set)", expanded=False):
            for name, m in RESEARCH_METRICS.items():
                star = " ⭐" if m.get("best") else ""
                st.markdown(
                    f"**{name}{star}**  \n"
                    f"Acc {m['accuracy']*100:.2f}% · "
                    f"F1 {m['f1']*100:.2f}% · "
                    f"AUC {m['auc']:.4f}"
                )

    return choice


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Vietnamese Fake News Detection",
        page_icon="📰",
        layout="wide",
    )

    # ---- Header --------------------------------------------------------
    st.markdown(
        "<h1 style='margin-bottom:4px;'>📰 Vietnamese Fake News Detection</h1>"
        "<p style='color:#6b7280;font-size:18px;margin-top:0;'>"
        "Machine Learning &amp; Deep Learning for Vietnamese News Classification</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    model_choice = render_sidebar()

    # ---- Two-column layout: input | results ----------------------------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Article input")
        raw_text = st.text_area(
            "Paste a Vietnamese news article below",
            height=320,
            placeholder=(
                "Ví dụ: Chính phủ vừa công bố chính sách mới về giáo dục. "
                "Theo đó, học sinh các trường tiểu học sẽ được miễn học phí từ "
                "năm học tới..."
            ),
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("🔍 Analyze", type="primary",
                                    use_container_width=True)

    with right:
        st.markdown("### Prediction")
        results_placeholder = st.container()

    # ---- Render research benchmark below -------------------------------
    st.markdown("---")
    render_research_benchmark()

    # ---- Run inference -------------------------------------------------
    if not analyze_clicked:
        st.info("Paste a Vietnamese article and press **Analyze** to see predictions.")
        return

    if not raw_text or not raw_text.strip():
        st.error("⚠️ Please paste an article before clicking **Analyze**.")
        return
    if len(raw_text.strip().split()) < 3:
        st.warning("The article is very short. Predictions may be unreliable.")

    clean_text = preprocess_text(raw_text)
    if not clean_text:
        st.error("The cleaned text is empty. Please provide more content.")
        return

    # Decide which model keys to run
    keys_to_run = MODEL_KEYS if model_choice == "all" else (model_choice,)
    results: dict[str, Prediction] = {}
    errors: dict[str, str] = {}

    with st.spinner("Running inference…"):
        for k in keys_to_run:
            try:
                results[k] = PREDICTORS[k](clean_text)
            except ArtifactMissingError as e:
                errors[k] = str(e)
            except Exception as e:  # pragma: no cover
                errors[k] = f"Unexpected error: {e}"

    # ---- Surface errors -------------------------------------------------
    if errors:
        with results_placeholder:
            st.error(
                "Some models could not be loaded. "
                "Train the corresponding model and run feature extraction first."
            )
            for k, msg in errors.items():
                st.warning(f"**{DISPLAY[k]}** — {msg}")

    if not results:
        st.error("No models produced a prediction. Check that artefacts exist on disk.")
        return

    # ---- Render results -------------------------------------------------
    with results_placeholder:
        if model_choice == "all":
            render_comparison(results)
        else:
            only = next(iter(results.values()))
            render_prediction(only)
            # Token count (if a deep model produced it)
            if only.n_tokens is not None:
                pass  # surfaced below in text statistics

    # ---- Text statistics ------------------------------------------------
    st.markdown("---")
    token_count = None
    if model_choice == "phobert" and "phobert" in results:
        token_count = results["phobert"].n_tokens
    elif model_choice == "bilstm" and "bilstm" in results:
        token_count = results["bilstm"].n_tokens
    render_text_stats(text_stats(raw_text, token_count=token_count))

    # ---- Disclaimer -----------------------------------------------------
    st.markdown("---")
    st.warning(
        "⚠️ **Research prototype.** "
        "Predictions come from models trained on a specific Vietnamese "
        "news corpus and should **not** be treated as definitive "
        "fact-checking. Do not rely on this demo to verify real-world news."
    )


if __name__ == "__main__":
    main()
