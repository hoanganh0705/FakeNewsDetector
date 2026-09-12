"""
Method-agreement analysis for token-level attributions (Phase 1, Step 1.6).

Two attribution methods rarely produce *identical* scores, so we quantify
how much they overlap on the *most-important tokens* rather than on raw
numerical values.  This module provides four primitives:

* ``rank_agreement``        — Jaccard overlap of the top-K tokens of two
  attribution vectors.
* ``faithfulness``          — "leave-one-out" style check: drop the top-K
  most-attributed tokens from the input and measure how much the model
  prediction moves.  This is the standard sanity check for
  attribution quality (Alvarez-Melis & Jaakkola, 2018).
* ``agreement_matrix``     — N×N pairwise rank-agreement heatmap between
  multiple methods on the same example.
* ``cross_model_agreement`` — intersection of the top-K most-attributed
  tokens across the *four* models of the system (LR, SVM, BiLSTM,
  PhoBERT).

Why agreement matters
---------------------
Token-level attributions are notoriously unstable across methods — IG,
SHAP and attention-rollout often disagree on *which* token is most
important.  Quantifying this disagreement is the headline finding that
the Phase-1 paper section turns into a table.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────


def rank_agreement(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    top_k: int = 10,
    tokens_a: Optional[Sequence[str]] = None,
    tokens_b: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Jaccard overlap of the top-K most-attributed tokens.

    The "most attributed" tokens are defined as the ``top_k`` indices
    with the largest *absolute* score, which is more meaningful than
    the largest signed score for binary classifiers (we want tokens
    that push the prediction either way).

    Args:
        scores_a: Attribution scores for the first method.
        scores_b: Attribution scores for the second method.
        top_k:   How many top tokens to consider per method.
        tokens_a: Optional token labels for ``scores_a`` (used to make
            the intersection readable).
        tokens_b: Optional token labels for ``scores_b``.

    Returns:
        Dict with ``jaccard`` (0..1), ``intersection`` and ``union``
        sizes, and the actual ``common_tokens`` (when ``tokens_a`` /
        ``tokens_b`` are provided).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"score vectors must have the same length "
            f"(got {len(scores_a)} vs {len(scores_b)})"
        )
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    top_a = _topk_indices(arr_a, top_k)
    top_b = _topk_indices(arr_b, top_k)

    # ``tokens_a`` is the *set of labels* to compare.  When the caller
    # passes labels we use them verbatim; otherwise we fall back to
    # integer indices so the Jaccard is well-defined and round-trip-safe.
    if tokens_a is None:
        tokens_a = list(range(len(arr_a)))
    if tokens_b is None:
        tokens_b = list(range(len(arr_b)))

    set_a = {tokens_a[int(i)] for i in top_a}
    set_b = {tokens_b[int(i)] for i in top_b}

    inter = set_a & set_b
    union = set_a | set_b
    jaccard = (len(inter) / len(union)) if union else 1.0

    return {
        "jaccard": float(jaccard),
        "intersection": int(len(inter)),
        "union": int(len(union)),
        "common_tokens": sorted(inter),
    }


def faithfulness(
    model_predict_proba: Callable[[str], np.ndarray],
    text: str,
    attribution: Sequence[float],
    remove_top_k: int = 5,
    mask_token: str = "",
) -> Dict[str, float]:
    """Drop the top-K most-attributed tokens; report the probability shift.

    A *faithful* attribution method should produce tokens whose removal
    causes the largest confidence drop for the predicted class.  We
    measure:

    * ``p_orig``     — predicted-class probability on the original input.
    * ``p_masked``   — predicted-class probability after masking the
      top-K tokens with ``mask_token`` (empty string by default).
    * ``drop``       — ``p_orig − p_masked`` (positive ⇒ faithful).

    Args:
        model_predict_proba: Callable mapping a raw string to a 2-vector
            of class probabilities (class-1 is the positive class).
        text: Original input string.
        attribution: Per-token attribution scores aligned with
            ``str.split()``.
        remove_top_k: How many top tokens to mask.
        mask_token: Replacement string for the masked tokens (default:
            empty string — i.e. token is removed).

    Returns:
        ``{p_orig, p_masked, drop, masked_tokens}`` dict.
    """
    words = str(text).split()
    if len(words) != len(attribution):
        raise ValueError(
            f"text/attribution length mismatch: {len(words)} vs {len(attribution)}"
        )
    if remove_top_k <= 0:
        raise ValueError("remove_top_k must be > 0")

    p_orig = np.asarray(model_predict_proba(text), dtype=np.float64)
    if p_orig.ndim == 1:
        p_orig = p_orig.reshape(1, -1)
    if p_orig.shape[-1] < 2:
        raise ValueError("predict_proba must return at least 2 classes")
    pred_class = int(np.argmax(p_orig[0]))
    p_orig_pred = float(p_orig[0, pred_class])

    arr = np.asarray(attribution, dtype=np.float64)
    top_idx = _topk_indices(arr, remove_top_k)
    masked_words = [w for i, w in enumerate(words) if i not in set(top_idx.tolist())]
    masked_text = (mask_token + " ").join(masked_words) + (f" {mask_token}" if masked_words else "")

    p_masked = np.asarray(model_predict_proba(masked_text), dtype=np.float64)
    p_masked_pred = float(p_masked[0, pred_class])
    drop = p_orig_pred - p_masked_pred

    return {
        "p_orig": float(p_orig_pred),
        "p_masked": float(p_masked_pred),
        "drop": float(drop),
        "pred_class": pred_class,
        "masked_tokens": [words[i] for i in top_idx],
    }


def agreement_matrix(
    methods_results: Mapping[str, Sequence[float]],
    top_k: int = 10,
    save_path: Optional[str] = None,
    title: str = "Rank-agreement between attribution methods",
) -> np.ndarray:
    """N×N pairwise Jaccard matrix between attribution methods.

    Args:
        methods_results: ``{method_name: scores_per_token}`` mapping.
            Score vectors may have different lengths (e.g. LR uses n-grams
            while BiLSTM/PhoBERT use word/subword tokenisation).  When
            lengths differ we skip that pair and leave the matrix cell
            as NaN so the heatmap still renders with a masked cell.
        top_k: Forwarded to :func:`rank_agreement`.
        save_path: Optional PNG path.  When given, the matrix is also
            rendered as a heatmap.
        title: Plot title.

    Returns:
        ``(N, N)`` ``np.ndarray`` of Jaccard scores in [0, 1] (NaN
        where a pair could not be compared).  The diagonal is 1.0.
    """
    names = list(methods_results.keys())
    n = len(names)
    matrix = np.eye(n, dtype=np.float64)
    nan_mask = np.zeros((n, n), dtype=bool)

    scores_list = [np.asarray(methods_results[k], dtype=np.float64) for k in names]
    len_list    = [len(s) for s in scores_list]

    for i in range(n):
        for j in range(i + 1, n):
            # Skip pairs with different tokenisation lengths.
            if len_list[i] != len_list[j]:
                matrix[i, j] = matrix[j, i] = np.nan
                nan_mask[i, j] = nan_mask[j, i] = True
                continue
            res = rank_agreement(scores_list[i], scores_list[j], top_k=top_k)
            matrix[i, j] = matrix[j, i] = res["jaccard"]

    if save_path:
        _render_heatmap(matrix, names, save_path, title, nan_mask)

    return matrix


def cross_model_agreement(
    text: str,
    lr_attrs: Optional[Sequence[float]] = None,
    svm_attrs: Optional[Sequence[float]] = None,
    bilstm_attrs: Optional[Sequence[float]] = None,
    phobert_attrs: Optional[Sequence[float]] = None,
    lr_tokens: Optional[List[str]] = None,
    svm_tokens: Optional[List[str]] = None,
    bilstm_tokens: Optional[List[str]] = None,
    phobert_tokens: Optional[List[str]] = None,
    top_k: int = 10,
    save_path: Optional[str] = None,
    title: str = "Cross-model agreement",
) -> Dict[str, object]:
    """Tokens on which the available models' top-K attribution lists agree.

    Uses the *native* tokenisation of each model (n-grams for LR/SVM,
    words for BiLSTM/PhoBERT) to find the Jaccard-overlap of top-K tokens
    within each *compatible pair* (LR vs SVM, BiLSTM vs PhoBERT).  When
    2+ models are available a grouped bar chart of the common tokens is
    rendered.

    Args:
        text: Original (word-segmented) input.
        lr_attrs / svm_attrs / bilstm_attrs / phobert_attrs: Per-token
            attribution scores (aligned with their respective ``*_tokens``).
        lr_tokens / svm_tokens / bilstm_tokens / phobert_tokens: Token
            strings that correspond to the score arrays.  Required for
            LR/SVM when their tokenisation differs from ``str.split(text)``.
        top_k: Top-K per model.
        save_path: Optional PNG path.
        title: Plot title.

    Returns:
        ``{"common_tokens", "available_models", "jaccard_pairs",
        "<model>_scores"}`` dict.
    """
    words = str(text).split()
    vectors: Dict[str, np.ndarray] = {}
    token_lists: Dict[str, List[str]] = {}

    if lr_attrs is not None:
        vectors["LR"]      = np.asarray(lr_attrs, dtype=np.float64)
        token_lists["LR"]  = lr_tokens if lr_tokens is not None else words
    if svm_attrs is not None:
        vectors["SVM"]     = np.asarray(svm_attrs, dtype=np.float64)
        token_lists["SVM"] = svm_tokens if svm_tokens is not None else words
    if bilstm_attrs is not None:
        vectors["BiLSTM"]  = np.asarray(bilstm_attrs, dtype=np.float64)
        token_lists["BiLSTM"] = bilstm_tokens if bilstm_tokens is not None else words
    if phobert_attrs is not None:
        vectors["PhoBERT"] = np.asarray(phobert_attrs, dtype=np.float64)
        token_lists["PhoBERT"] = phobert_tokens if phobert_tokens is not None else words

    if not vectors:
        raise ValueError("At least one model's attribution vector must be provided")

    # ── Per-model top-K using *native* tokenisation ──────────────────
    tops: Dict[str, set[str]] = {}
    for name, vec in vectors.items():
        idx = _topk_indices(vec, top_k)
        tops[name] = {str(i) for i in idx}   # store integer indices

    # ── Jaccard per compatible pair ───────────────────────────────────
    jaccard_pairs: Dict[str, float] = {}
    for (na, ta), (nb, tb) in [
        (("LR", tops.get("LR", {})), ("SVM", tops.get("SVM", {}))),
        (("BiLSTM", tops.get("BiLSTM", {})), ("PhoBERT", tops.get("PhoBERT", {}))),
    ]:
        if ta and tb:
            inter = ta & tb
            union = ta | tb
            jaccard_pairs[f"{na}_vs_{nb}"] = len(inter) / len(union) if union else 1.0

    result: Dict[str, object] = {
        "available_models":  list(vectors.keys()),
        "jaccard_pairs":     jaccard_pairs,
    }

    # ── Bar chart: common tokens across *all* available models ────────
    if len(vectors) >= 2:
        # Align all models to word level before computing intersection.
        word_vecs: Dict[str, np.ndarray] = {}
        for name in vectors:
            vec  = vectors[name]
            toks = token_lists[name]
            if len(vec) == len(words) == len(toks):
                word_vecs[name] = vec
            else:
                word_vecs[name] = _align_to_words_simple(words, toks, vec)

        word_tops: Dict[str, set[int]] = {}
        for name, vec in word_vecs.items():
            idx = _topk_indices(vec, top_k)
            word_tops[name] = set(idx)

        common_ints = set.intersection(*word_tops.values()) if word_tops else set()
        common = [words[i] for i in sorted(common_ints)]
        result["common_tokens"] = common

        if save_path:
            note = "(No common top-K tokens)" if not common else ""
            _render_cross_model_bars(
                common, word_vecs, words, save_path, title, note=note,
            )

    return result


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest-absolute elements of ``arr``.

    When there are ties (multiple elements with the same absolute
    value), ``np.argpartition`` may return more than ``k`` indices
    because the partition is only guaranteed to place the cut *between*
    the top-k and the rest.  We fix the length by taking the first
    ``k`` indices returned by ``argsort`` (descending), which is also
    O(n log n) in the worst case but unambiguous.
    """
    if k >= len(arr):
        return np.arange(len(arr))
    abs_arr = np.abs(arr)
    # ``argsort`` is O(n log n) but the descending-absolute order is
    # unambiguous; we then slice the top-k.
    order = np.argsort(-abs_arr, kind="stable")
    return order[:k]


def _render_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    save_path: str,
    title: str,
    nan_mask: Optional[np.ndarray] = None,
) -> None:
    """Save an N×N heatmap to ``save_path`` (PNG + PDF).

    NaN cells are rendered as grey ("0.5" grayscale) and labelled "—".
    """
    import os

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, 0.8 * n + 2), max(4, 0.8 * n + 2)))

    # Replace NaN with a placeholder value for imshow (grey = 0.5).
    display = matrix.copy()
    display[np.isnan(display)] = 0.5

    im = ax.imshow(display, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if nan_mask is not None and nan_mask[i, j]:
                ax.text(j, i, "—", ha="center", va="center",
                        color="gray", fontsize=10)
            elif np.isnan(matrix[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        color="gray", fontsize=10)
            else:
                colour = "white" if matrix[i, j] < 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color=colour, fontsize=10)

    fig.colorbar(im, ax=ax, label="Jaccard(top-10)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _render_cross_model_bars(
    common: set,
    vectors: Dict[str, np.ndarray],
    words: List[str],
    save_path: str,
    title: str,
    note: str = "",
) -> None:
    """Grouped bar chart — each token's |attribution| per model.

    Works with any subset of models (2–4).  Models not in ``vectors``
    are simply omitted from the chart.  When ``common`` is empty an
    informative placeholder figure is still rendered (not a blank PNG).
    """
    import os

    tokens_sorted = sorted(common)
    n_models = len(vectors)
    fig_height = max(2.0, 0.45 * len(tokens_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    if tokens_sorted and n_models > 0:
        width = 0.2
        positions = np.arange(len(tokens_sorted))
        colours = {
            "LR":      "#1f77b4",
            "SVM":     "#ff7f0e",
            "BiLSTM":  "#2ca02c",
            "PhoBERT": "#d62728",
        }
        ordered = [k for k in ("LR", "SVM", "BiLSTM", "PhoBERT") if k in vectors]
        offsets = np.linspace(-(n_models - 1) * width / 2,
                               (n_models - 1) * width / 2,
                               n_models)

        for i, name in enumerate(ordered):
            vec = vectors[name]
            vals = [abs(float(vec[words.index(t)])) for t in tokens_sorted]
            ax.barh(positions + offsets[i], vals, height=width,
                    label=name, color=colours.get(name, f"C{i}"), alpha=0.85)

        ax.set_yticks(positions)
        ax.set_yticklabels(tokens_sorted, fontsize=10)
        ax.set_xlabel("|Attribution| (model-specific scale)", fontsize=11)
    else:
        # Empty chart — still render something informative.
        ax.text(0.5, 0.5,
                f"No common top-K tokens\nacross the available models.\n{note}",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=13, fontweight="bold")
    if n_models > 0 and tokens_sorted:
        ax.legend(loc="lower right", fontsize=9)
        ax.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _align_to_words_simple(
    words: List[str],
    toks: List[str],
    scores: np.ndarray,
) -> np.ndarray:
    """Align a native-token score vector to word level.

    ``toks`` and ``scores`` have the same length (one entry per TF-IDF
    n-gram token).  We split each n-gram on ``_`` and distribute its
    score equally to every component word.  Returns a ``(len(words),)``
    vector.
    """
    word_scores = np.zeros(len(words), dtype=np.float64)
    counts      = np.zeros(len(words), dtype=np.float64)
    word_norm  = [w.lower().replace("_", " ") for w in words]

    for tok, sc in zip(toks, scores):
        parts = _split_token_ngram(str(tok))
        for part in parts:
            part_lc = part.lower().replace("_", " ")
            if not part_lc:
                continue
            for wi, wn in enumerate(word_norm):
                if part_lc == wn or wn.startswith(part_lc) or part_lc.startswith(wn):
                    word_scores[wi] += sc
                    counts[wi]      += 1.0

    counts = np.where(counts == 0, 1, counts)
    return word_scores / counts


def _split_token_ngram(tok: str) -> List[str]:
    """Split an n-gram token into component words (underscore-separated)."""
    tok = tok.replace("▁", "").replace("##", "")
    return [p for p in tok.split("_") if p.strip()]


__all__ = [
    "rank_agreement",
    "faithfulness",
    "agreement_matrix",
    "cross_model_agreement",
]