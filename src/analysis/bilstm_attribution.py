"""
Token-level attribution for the BiLSTM classifier.

Implements two gradient-based methods from ``IMPLEMENTATION_PLAN.md`` §5,
Step 1.4:

* ``bilstm_simple_gradients`` — vanilla saliency: the gradient of the
  positive-class logit w.r.t. the embedding output, summarised per token by
  the L2 norm across embedding dimensions.
* ``bilstm_ig``             — Integrated Gradients (Sundararajan et al.,
  2017).  IG is defined as the path integral of gradients from a zero
  embedding (the natural baseline for the BiLSTM, where every non-pad
  token receives a non-zero gradient) to the actual embedding.

Both helpers take the same arguments and return aligned
``(tokens, scores)`` arrays, which the visualization helper renders as a
red/blue heatmap (positive = pushes towards Fake).

Notes
-----
* ``model`` is the raw ``BiLSTMClassifier`` (from
  ``src.models.bilstm_model``).  The wrapper ``BiLSTMTrainer`` exposes it
  via ``trainer.model``.
* ``vocab`` is either an ``EmbeddingFeatureExtractor`` (preferred) or a
  raw ``Vocabulary`` instance with ``word2idx`` / ``idx2word``.  We
  support both because downstream code may pass either one.
* All gradients are computed on CPU unless ``model`` itself lives on a
  CUDA device — we honour that device.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────


def bilstm_simple_gradients(
    text: str,
    model,
    vocab,
    max_len: int = 128,
    target_class: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """Vanilla gradient saliency for a BiLSTM model.

    The saliency score for each token is ``||∂y_target/∂e_i||_2``, where
    ``e_i`` is the embedding vector of the i-th token.  This is a fast,
    deterministic baseline against which to compare Integrated Gradients.

    Args:
        text: Word-segmented Vietnamese document.
        model: A ``BiLSTMClassifier`` (raw ``nn.Module``, not the trainer).
        vocab: Vocabulary or ``EmbeddingFeatureExtractor`` exposing
            ``word2idx`` / ``idx2word``.
        max_len: Maximum sequence length; longer documents are truncated.
        target_class: Which output neuron to explain (1 for "Fake").

    Returns:
        ``(tokens, scores)`` — list of token strings and a matching array
        of saliency magnitudes (non-negative).
    """
    idx2word, word2idx = _resolve_vocab(vocab)
    device = _device_of(model)

    token_ids, tokens = _encode_text(text, word2idx, idx2word, max_len)
    if not token_ids:
        return [], np.array([], dtype=np.float64)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    embedding = model.embedding  # (V, E)
    embeds = embedding(input_ids).detach().clone().requires_grad_(True)

    mask = (input_ids != 0).long()
    logits = model.lstm(embeds)  # placeholder — we need the actual path

    # Mirror the forward pass to keep attributions aligned with the model's
    # real computation graph (masked mean-pool + dropout + classifier head).
    lstm_out, _ = model.lstm(embeds)
    if mask is not None:
        m = mask.to(lstm_out.dtype).unsqueeze(-1)
        pooled = (lstm_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    else:
        pooled = lstm_out.mean(dim=1)
    pooled = model.dropout(pooled)
    logits = model.classifier(pooled)
    target_logit = logits[0, target_class]

    grad = torch.autograd.grad(target_logit, embeds, retain_graph=False)[0]
    saliency = grad[0].detach().cpu().norm(p=2, dim=-1).numpy()

    return tokens, saliency.astype(np.float64)


def bilstm_ig(
    text: str,
    model,
    vocab,
    max_len: int = 128,
    n_steps: int = 50,
    target_class: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """Integrated Gradients (Sundararajan et al., 2017) for a BiLSTM model.

    We integrate gradients along the straight-line path from the
    zero-embedding baseline ``x'=0`` to the actual embedding ``x`` with
    ``n_steps`` Riemann samples.  Each token's IG value is the L2 norm
    (over the embedding dimension) of the integrated gradient, matching
    the convention used by the simple-gradients helper so the same
    visualisation code works for both.

    Args:
        text: Word-segmented Vietnamese document.
        model: ``BiLSTMClassifier`` instance.
        vocab: Vocabulary / extractor with ``word2idx`` / ``idx2word``.
        max_len: Maximum sequence length (truncate longer inputs).
        n_steps: Number of integration steps (Riemann sum).
        target_class: Output index to explain (1 = "Fake").

    Returns:
        ``(tokens, scores)`` aligned 1-to-1 with the *non-pad* tokens.
    """
    idx2word, word2idx = _resolve_vocab(vocab)
    device = _device_of(model)

    token_ids, tokens = _encode_text(text, word2idx, idx2word, max_len)
    if not token_ids:
        return [], np.array([], dtype=np.float64)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    embedding = model.embedding
    with torch.no_grad():
        x_embed = embedding(input_ids).clone()  # (1, T, E)
    baseline = torch.zeros_like(x_embed)

    # Trapezoidal Riemann sum along [0, 1] in n_steps+1 segments.
    mask = (input_ids != 0).long()

    accumulated = torch.zeros_like(x_embed)
    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        interpolated = baseline + alpha * (x_embed - baseline)
        interpolated = interpolated.detach().requires_grad_(True)

        lstm_out, _ = model.lstm(interpolated)
        m = mask.to(lstm_out.dtype).unsqueeze(-1)
        pooled = (lstm_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        pooled = model.dropout(pooled)
        logits = model.classifier(pooled)
        target_logit = logits[0, target_class]
        grad = torch.autograd.grad(target_logit, interpolated)[0]

        accumulated = accumulated + grad

    avg_grad = accumulated / float(n_steps)
    ig = (x_embed - baseline) * avg_grad  # element-wise
    ig_per_token = ig[0].detach().cpu().norm(p=2, dim=-1).numpy()

    return tokens, ig_per_token.astype(np.float64)


def visualize_bilstm_attribution(
    text: str,
    attribution: Sequence[float],
    save_path: str,
    title: str = "BiLSTM attribution",
    cmap_pos: str = "#d62728",
    cmap_neg: str = "#1f77b4",
) -> str:
    """Render a colour-coded token heatmap.

    The visualisation mirrors the SHAP chart so the reader can compare
    methods at a glance.  Tokens are coloured by sign of the attribution:

    * red  (``#d62728``) — positive contribution to the Fake class
    * blue (``#1f77b4``) — negative contribution (pushes towards Real)

    Args:
        text: Original word-segmented document.  Must align 1-to-1 with
            ``attribution``.
        attribution: One float per non-pad token.
        save_path: Destination PNG path.  Directory is created if missing.
        title: Figure title.
        cmap_pos / cmap_neg: Hex colours for the two signs.

    Returns:
        The PNG path that was written.
    """
    tokens = str(text).split()
    if len(tokens) != len(attribution):
        raise ValueError(
            f"text/attribution length mismatch: {len(tokens)} vs {len(attribution)}"
        )
    if len(tokens) == 0:
        raise ValueError("Cannot visualise empty attribution.")

    scores = np.asarray(attribution, dtype=np.float64)
    abs_max = float(np.max(np.abs(scores))) if np.any(scores != 0) else 1.0

    # Build a horizontal "token" strip where the background colour of each
    # cell encodes the attribution value.  Padding tokens are dimmed.
    fig_height = max(1.6, 0.32 * len(tokens) + 1.2)
    fig, ax = plt.subplots(figsize=(min(14, 0.55 * len(tokens) + 2), fig_height))
    ax.set_axis_off()

    cell_h = 1.0
    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        if sc >= 0:
            colour = cmap_pos
        else:
            colour = cmap_neg
        # Saturation proportional to |sc| / abs_max
        alpha = 0.25 + 0.75 * (abs(sc) / abs_max if abs_max > 0 else 0.0)
        rect = plt.Rectangle(
            (i, 0), 1.0, cell_h,
            facecolor=colour, alpha=alpha, edgecolor="white", linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.5, cell_h / 2, tok,
            ha="center", va="center",
            fontsize=10,
            color="black" if alpha < 0.55 else "white",
        )

    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, cell_h)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Legend
    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(color=cmap_pos, label="Pushes → Fake"),
            Patch(color=cmap_neg, label="Pushes → Real"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
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


def _resolve_vocab(vocab) -> Tuple[dict, dict]:
    """Return ``(idx2word, word2idx)`` regardless of which vocab object was passed.

    Accepts, in order:

    * a raw ``Vocabulary`` with ``.idx2word`` / ``.word2idx`` attributes,
    * an ``EmbeddingFeatureExtractor`` whose ``.vocab`` attribute is such a Vocabulary,
    * the ``joblib.load``-style *dict* produced by ``EmbeddingFeatureExtractor.save``
      (i.e. ``{"vocab": <Vocabulary>, "max_vocab_size": ...}``).
    """
    # Case 3: a plain dict saved by Extractor.save() — unpack the vocab.
    if isinstance(vocab, dict):
        if "vocab" in vocab and hasattr(vocab["vocab"], "idx2word"):
            vocab = vocab["vocab"]
        elif "idx2word" in vocab and "word2idx" in vocab:
            # Already in the canonical dict form.
            return vocab["idx2word"], vocab["word2idx"]

    # Case 2: an Extractor wrapper.
    if hasattr(vocab, "vocab") and hasattr(vocab.vocab, "idx2word"):
        vocab = vocab.vocab

    idx2word = vocab.idx2word
    word2idx = vocab.word2idx
    return idx2word, word2idx


def _encode_text(
    text: str,
    word2idx: dict,
    idx2word: dict,
    max_len: int,
) -> Tuple[List[int], List[str]]:
    """Convert ``text`` to a list of token ids (truncated)."""
    words = str(text).split()
    if not words:
        return [], []
    if len(words) > max_len:
        words = words[:max_len]

    unk_idx = word2idx.get("<UNK>", 1)
    token_ids: List[int] = [word2idx.get(w, unk_idx) for w in words]
    return token_ids, words


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


__all__ = [
    "bilstm_simple_gradients",
    "bilstm_ig",
    "visualize_bilstm_attribution",
]