from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap



def lr_kernel_shap(
    text: str,
    model,
    vectorizer,
    n_samples: int = 200,
) -> Tuple[List[str], np.ndarray]:

    feature_names = _feature_names(vectorizer)

    background = np.zeros((1, len(feature_names)), dtype=np.float32)

    def _logit_fn(X):
        return model.decision_function(X)

    explainer = shap.KernelExplainer(
        _logit_fn,
        background,
        silent=True,
    )

    x = vectorizer.transform([text])
    sv = explainer.shap_values(x, nsamples=n_samples, silent=True)

    sv = np.asarray(sv).reshape(-1)

    x_dense = np.asarray(x.todense()).reshape(-1)

    tokens, scores = _collect_present_tokens(feature_names, x_dense, sv)
    return tokens, np.asarray(scores, dtype=np.float64)


def svm_linear_shap(
    text: str,
    model,
    vectorizer,
) -> Tuple[List[str], np.ndarray]:
    feature_names = _feature_names(vectorizer)

    base = _unwrap_linear_svc(model)
    if base is None:
        return lr_kernel_shap(text, model, vectorizer, n_samples=200)

    x = vectorizer.transform([text])
    x_dense = np.asarray(x.todense()).reshape(-1)

    w = np.asarray(base.coef_).reshape(-1)
    sv = x_dense * w

    tokens, scores = _collect_present_tokens(feature_names, x_dense, sv)
    return tokens, np.asarray(scores, dtype=np.float64)


def visualize_token_importance(
    tokens: Sequence[str],
    scores: Sequence[float],
    save_path: str,
    top_k: int = 15,
    title: str = "Token attribution",
) -> str:
    if len(tokens) != len(scores):
        raise ValueError(
            f"tokens/scores length mismatch: {len(tokens)} vs {len(scores)}"
        )
    if len(tokens) == 0:
        raise ValueError("Cannot visualise empty token list.")

    order = np.argsort(np.abs(scores))[::-1][: max(1, int(top_k))]
    sel_tokens = [tokens[i] for i in order]
    sel_scores = np.asarray([scores[i] for i in order], dtype=np.float64)

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


def _feature_names(vectorizer) -> List[str]:
    names = vectorizer.get_feature_names_out()
    return [str(n) for n in names]


def _collect_present_tokens(
    feature_names: Sequence[str],
    x_dense: np.ndarray,
    shap_values: np.ndarray,
) -> Tuple[List[str], List[float]]:
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
    for attr in ("estimator", "base_estimator"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "coef_"):
            return inner
    if hasattr(model, "coef_"):
        return model
    return None


__all__ = [
    "lr_kernel_shap",
    "svm_linear_shap",
    "visualize_token_importance",
]