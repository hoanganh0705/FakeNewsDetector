from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def bilstm_simple_gradients(
    text: str,
    model,
    vocab,
    max_len: int = 128,
    target_class: int = 1,
) -> Tuple[List[str], np.ndarray]:
    idx2word, word2idx = _resolve_vocab(vocab)
    device = _device_of(model)

    token_ids, tokens = _encode_text(text, word2idx, idx2word, max_len)
    if not token_ids:
        return [], np.array([], dtype=np.float64)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    embedding = model.embedding 
    embeds = embedding(input_ids).detach().clone().requires_grad_(True)

    mask = (input_ids != 0).long()
    logits = model.lstm(embeds)

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
    idx2word, word2idx = _resolve_vocab(vocab)
    device = _device_of(model)

    token_ids, tokens = _encode_text(text, word2idx, idx2word, max_len)
    if not token_ids:
        return [], np.array([], dtype=np.float64)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    embedding = model.embedding
    with torch.no_grad():
        x_embed = embedding(input_ids).clone()
    baseline = torch.zeros_like(x_embed)

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
    tokens = str(text).split()
    if len(tokens) != len(attribution):
        raise ValueError(
            f"text/attribution length mismatch: {len(tokens)} vs {len(attribution)}"
        )
    if len(tokens) == 0:
        raise ValueError("Cannot visualise empty attribution.")

    scores = np.asarray(attribution, dtype=np.float64)
    abs_max = float(np.max(np.abs(scores))) if np.any(scores != 0) else 1.0

    fig_height = max(1.6, 0.32 * len(tokens) + 1.2)
    fig, ax = plt.subplots(figsize=(min(14, 0.55 * len(tokens) + 2), fig_height))
    ax.set_axis_off()

    cell_h = 1.0
    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        if sc >= 0:
            colour = cmap_pos
        else:
            colour = cmap_neg
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


def _resolve_vocab(vocab) -> Tuple[dict, dict]:

    if isinstance(vocab, dict):
        if "vocab" in vocab and hasattr(vocab["vocab"], "idx2word"):
            vocab = vocab["vocab"]
        elif "idx2word" in vocab and "word2idx" in vocab:
            return vocab["idx2word"], vocab["word2idx"]

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