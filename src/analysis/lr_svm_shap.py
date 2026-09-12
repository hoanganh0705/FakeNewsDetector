"""
Token-level SHAP attribution for the linear (LR) and linear-SVM models.

SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) assigns each input
feature a contribution value such that ``sum(shap_values) ≈ model_output −
expected_value``.  For a TF-IDF classifier the input "tokens" are the n-grams
present in the document; for SHAP we still have to materialise a dense or
sparse feature vector per document before calling the explainer.

This module is part of Phase 1 (Explainability) of the implementation plan
(see ``IMPLEMENTATION_PLAN.md`` §5, steps 1.3).  It is intentionally
self-contained: the helpers can be imported and reused by the orchestrator
in ``src/analysis/explainability_runner.py`` and the tests in
``tests/test_explainability_attribution.py``.

Public API
----------
* ``lr_kernel_shap(text, model, vectorizer, n_samples=100)``     → (tokens, shap_values)
* ``svm_linear_shap(text, model, vectorizer)``                   → (tokens, shap_values)
* ``visualize_token_importance(tokens, scores, save_path, top_k=15)``
    Horizontal bar chart; red bars = positive class (Fake), blue = real.

Notes
-----
* ``shap`` is imported lazily inside the helpers so that the rest of the
  package can still be imported on systems where the (heavy) SHAP stack
  is not installed (e.g. during a quick CI sanity check).
* ``model`` is expected to expose ``predict_proba`` / ``decision_function``
  in the usual sklearn fashion.  The SVM variant is calibrated with a
  ``CalibratedClassifierCV`` wrapper during training, so we always go
  through ``predict_proba`` there.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for batch figure generation
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────


def lr_kernel_shap(
    text: str,
    model,
    vectorizer,
    n_samples: int = 200,
) -> Tuple[List[str], np.ndarray]:
    """Compute KernelSHAP attributions for a Logistic-Regression TF-IDF model.

    SHAP requires a *background dataset* to estimate the missing-feature
    baseline.  For sparse, high-dimensional TF-IDF vectors we use a small
    zero-vector background (``shap.kmeans`` is unsuitable here because the
    sparse representation has thousands of features and few informative
    ones per document).  ``n_samples`` controls how many coalitions SHAP
    samples per explanation.

    Args:
        text: A *single* Vietnamese document (already word-segmented; the
            vectorizer expects its own token format).
        model: Trained sklearn classifier (e.g. ``LogisticRegression``).
        vectorizer: Fitted ``TfidfVectorizer`` used to featurise the model.
        n_samples: Number of feature-coalition samples for the KernelSHAP
            estimator.  200 is a good speed/quality trade-off for vocab
            sizes of ~40k (sparse, high-dim data needs more samples).

    Returns:
        ``(tokens, shap_values)`` where

        * ``tokens``       is the list of token strings actually present
                            in ``text`` (deduplicated, in document order),
        * ``shap_values`` is a ``np.ndarray`` of the same length giving
                            each token's contribution to the *positive*
                            class (Fake = 1) in log-odds space.
    """
    import shap  # lazy — keeps import cost off the cold path

    feature_names = _feature_names(vectorizer)

    # Single dense background vector of all zeros — the natural "absent
    # token" reference for TF-IDF (the IDF weights imply missing tokens
    # should contribute 0 to the linear model).
    background = np.zeros((1, len(feature_names)), dtype=np.float32)

    # KernelSHAP works best with a *raw* model output (logits for LR, margin
    # for SVM).  Passing ``predict_proba`` with ``link="logit"`` is a known
    # foot-gun: SHAP treats the probability as if it were a log-odds value
    # and rescales the attributions by ``p·(1-p)``, which crushes them to
    # ~1e-4 when probabilities are extreme — making the bar chart look
    # empty.  ``decision_function`` returns the actual log-odds directly
    # so we drop the ``link`` argument entirely.
    def _logit_fn(X):
        # ``model.decision_function`` returns (N,) for binary LR; SHAP
        # expects (N, 1) for the positive class explanation.
        return model.decision_function(X)

    explainer = shap.KernelExplainer(
        _logit_fn,
        background,
        silent=True,
    )

    # Featurise the single text and ask SHAP to explain it.
    x = vectorizer.transform([text])
    sv = explainer.shap_values(x, nsamples=n_samples, silent=True)

    # With a single-output ``_logit_fn`` SHAP returns one array of shape
    # (1, V).  Flatten to (V,) so the rest of the pipeline sees the
    # canonical layout it expects.
    sv = np.asarray(sv).reshape(-1)

    # ``x`` is sparse (1, V); collapse to dense row 0.
    x_dense = np.asarray(x.todense()).reshape(-1)

    tokens, scores = _collect_present_tokens(feature_names, x_dense, sv)
    return tokens, np.asarray(scores, dtype=np.float64)


def svm_linear_shap(
    text: str,
    model,
    vectorizer,
) -> Tuple[List[str], np.ndarray]:
    """Compute exact per-token attributions for a LinearSVC-based classifier.

    Linear SVMs sitting behind a ``CalibratedClassifierCV`` wrapper are
    not natively supported by ``shap.LinearExplainer`` because:
      (a) the wrapper is non-linear, and
      (b) ``LinearExplainer.__init__`` cannot fit a covariance model on
          a single-row background (which is the natural TF-IDF baseline).

    We sidestep both issues by computing the *exact* SHAP values
    directly from the uncalibrated ``LinearSVC`` weights: for an
    additive linear model with a zero-valued baseline (the only
    meaningful reference for sparse TF-IDF), the Shapley value of
    feature ``i`` reduces to ``w_i · x_i``.  This is what
    ``LinearExplainer`` returns internally for ``feature_perturbation=
    "interventional"`` on a wide-enough background, so we just call the
    math ourselves — it is O(V) and deterministic.

    Args:
        text: A single document.
        model: Trained classifier (CalibratedClassifierCV or LinearSVC).
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        ``(tokens, shap_values)`` — same shape contract as
        :func:`lr_kernel_shap`.
    """
    feature_names = _feature_names(vectorizer)

    base = _unwrap_linear_svc(model)
    if base is None:
        # Should never happen in practice — fall back to a (slower) KernelSHAP
        return lr_kernel_shap(text, model, vectorizer, n_samples=200)

    # Closed-form SHAP: ``phi_i = w_i · x_i`` for additive models with a
    # zero baseline.  No covariance, no SHAP explainer needed.
    x = vectorizer.transform([text])
    x_dense = np.asarray(x.todense()).reshape(-1)

    # ``base.coef_`` has shape (1, V) for binary LinearSVC.
    w = np.asarray(base.coef_).reshape(-1)
    sv = x_dense * w  # element-wise -> (V,)

    tokens, scores = _collect_present_tokens(feature_names, x_dense, sv)
    return tokens, np.asarray(scores, dtype=np.float64)


def visualize_token_importance(
    tokens: Sequence[str],
    scores: Sequence[float],
    save_path: str,
    top_k: int = 15,
    title: str = "Token attribution",
) -> str:
    """Render a horizontal bar chart of the most influential tokens.

    Bars are coloured by sign:
    * red  (``#d62728``) — pushes the prediction towards the positive class (Fake)
    * blue (``#1f77b4``) — pushes towards the negative class (Real)

    Args:
        tokens: Token strings (must align 1-to-1 with ``scores``).
        scores: Attribution values (log-odds contribution per token).
        save_path: Destination ``.png`` path.  The directory is created
            if it does not exist.  The same path with ``.pdf`` extension
            is also written for LaTeX.
        top_k: Keep only the ``top_k`` tokens with the largest *absolute*
            score.
        title: Figure title.

    Returns:
        The path that was written (PNG).
    """
    if len(tokens) != len(scores):
        raise ValueError(
            f"tokens/scores length mismatch: {len(tokens)} vs {len(scores)}"
        )
    if len(tokens) == 0:
        raise ValueError("Cannot visualise empty token list.")

    # Sort by absolute score; keep top_k extremes on both sides.
    order = np.argsort(np.abs(scores))[::-1][: max(1, int(top_k))]
    sel_tokens = [tokens[i] for i in order]
    sel_scores = np.asarray([scores[i] for i in order], dtype=np.float64)

    # Display tokens with the largest *positive* score at the top.
    sort_idx = np.argsort(sel_scores)
    sel_tokens = [sel_tokens[i] for i in sort_idx]
    sel_scores = sel_scores[sort_idx]

    fig_height = max(2.5, 0.32 * len(sel_tokens) + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    colors = ["#d62728" if s > 0 else "#1f77b4" for s in sel_scores]
    ax.barh(range(len(sel_tokens)), sel_scores, color=colors, alpha=0.85)

    ax.set_yticks(range(len(sel_tokens)))
    ax.set_yticklabels(sel_tokens, fontsize=10)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP value (log-odds contribution)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Legend (manual so the colour/sign mapping is explicit)
    from matplotlib.patches import Patch
    legend = ax.legend(
        handles=[
            Patch(color="#d62728", label="Pushes → Fake"),
            Patch(color="#1f77b4", label="Pushes → Real"),
        ],
        loc="lower right",
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _feature_names(vectorizer) -> List[str]:
    """Return the vectorizer's feature names as a plain list[str]."""
    names = vectorizer.get_feature_names_out()
    return [str(n) for n in names]


def _collect_present_tokens(
    feature_names: Sequence[str],
    x_dense: np.ndarray,
    shap_values: np.ndarray,
) -> Tuple[List[str], List[float]]:
    """Return ``(tokens, scores)`` for tokens whose TF-IDF weight is > 0.

    Different tokens can map to the same n-gram index when the document
    contains repeated words; we deduplicate by keeping the *maximum*
    absolute SHAP value per token to avoid spurious bar duplicates.
    """
    present = np.where(x_dense > 0)[0]
    if present.size == 0:
        return [], []

    best_per_token: dict[str, float] = {}
    for idx in present:
        tok = feature_names[idx]
        score = float(shap_values[idx])
        prev = best_per_token.get(tok)
        if prev is None or abs(score) > abs(prev):
            best_per_token[tok] = score

    # Preserve document order — SHAP is order-agnostic, but humans read
    # left-to-right and the chart is easier to interpret that way.
    seen: set[str] = set()
    tokens: List[str] = []
    scores: List[float] = []
    for idx in present:
        tok = feature_names[idx]
        if tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
        scores.append(best_per_token[tok])
    return tokens, scores


def _unwrap_linear_svc(model):
    """Return the underlying ``LinearSVC`` if *model* is a calibrated wrapper."""
    # CalibratedClassifierCV stores the base estimator under ``estimator``
    # (sklearn ≥ 1.2) or ``base_estimator`` (older versions).
    for attr in ("estimator", "base_estimator"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "coef_"):
            return inner
    if hasattr(model, "coef_"):
        return model
    return None


# ──────────────────────────────────────────────────────────────────────
# Public-all export list
# ──────────────────────────────────────────────────────────────────────


__all__ = [
    "lr_kernel_shap",
    "svm_linear_shap",
    "visualize_token_importance",
]