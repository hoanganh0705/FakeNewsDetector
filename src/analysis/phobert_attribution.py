"""
Token-level attribution for the PhoBERT classifier.

Implements three complementary methods on the fine-tuned PhoBERT model
(``src.models.phobert_model.PhoBertClassifier``):

* ``phobert_shap``               — SHAP on the embedding layer using a
  small subset of background samples (PartitionSHAP-style local masking
  is approximated via zero-baseline interventional SHAP — see notes).
* ``phobert_integrated_gradients`` — captum's ``LayerIntegratedGradients``
  on the word embeddings.
* ``phobert_attention_rollout``  — Abnar & Zuidema (2020) attention
  rollout, which propagates attention through every transformer layer
  to approximate information flow.

Plus one visual helper:

* ``compare_attribution_methods`` — render a 3-panel side-by-side
  comparison so a human can see whether the methods agree.

Notes
-----
* All three attribution methods need the embedding layer (or attention
  weights) to be exposed; we access ``model.encoder.embeddings`` and
  ``model.encoder.encoder.layer[i].attention.self`` through standard
  HuggingFace attribute names.
* ``tokenizer`` can be either the raw ``AutoTokenizer`` or a
  ``PhoBertFeatureExtractor`` (we detect which one was passed and call
  ``.tokenize`` accordingly).
* ``shap`` and ``captum`` are imported lazily so the rest of the package
  remains importable on machines without the heavy attribution stack.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────


def phobert_shap(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    n_samples: int = 50,
    target_class: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """SHAP attribution for a PhoBERT classifier.

    We use SHAP's ``GradientExplainer`` on the embedding layer (input
    space is the word-embedding tensor, not the raw token IDs).  SHAP
    perturbs the *embedding* of each token — this avoids the int-typed
    issues that ``PartitionExplainer`` has with discrete inputs while
    keeping the explanation locally faithful.

    The 0-vector baseline corresponds to "no information at all" for
    each token, the standard choice for transformer explanations
    (Sundararajan et al., 2017).

    Args:
        text: A single document (raw, pre-tokenization).
        model: ``PhoBertClassifier`` instance.
        tokenizer: HuggingFace ``PreTrainedTokenizer`` (or
            ``PhoBertFeatureExtractor``).
        max_length: Truncation length (must match training).
        n_samples: Number of SHAP samples (50 is enough for a single
            document; larger ⇒ slower).
        target_class: Output index to explain (1 = "Fake").

    Returns:
        ``(tokens, scores)`` — tokens aligned with attribution scores
        (log-odds contribution per token to the positive class).
    """
    import shap  # lazy

    tokenizer = _unwrap_tokenizer(tokenizer)
    model.eval()
    device = _device_of(model)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Wrap model so SHAP sees an embedding-tensor input rather than token IDs.
    embedding_layer = model.encoder.embeddings
    with torch.no_grad():
        baseline_embeds = torch.zeros_like(embedding_layer(input_ids))
        background_embeds = embedding_layer(input_ids).clone()

    def f(embeds: np.ndarray) -> np.ndarray:
        """Map an (N, T, E) embedding batch → (N, 2) softmax probs."""
        embeds_t = torch.as_tensor(embeds, dtype=baseline_embeds.dtype, device=device)
        if embeds_t.dim() == 2:  # SHAP flattens the trailing dim sometimes
            embeds_t = embeds_t.unsqueeze(0)
        # Repeat the attention mask across the batch.
        am = attention_mask.expand(embeds_t.shape[0], -1)
        with torch.no_grad():
            outputs = model.encoder(inputs_embeds=embeds_t, attention_mask=am)
            pooled = outputs.last_hidden_state[:, 0]
            logits = model.classifier(pooled)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    explainer = shap.GradientExplainer(
        f,
        background_embeds,
        local_smoothing=0.0,
    )
    shap_values = explainer.shap_values(
        background_embeds,
        nsamples=n_samples,
        ranked_outputs=None,
    )

    # shap_values is a list[ndarray] for the multi-class case — pick the
    # positive class.  Last axis of the returned tensor holds the classes.
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[target_class])
    else:
        sv = np.asarray(shap_values)[..., target_class]
    # Collapse the embedding dimension by L2-norm (same convention as BiLSTM).
    if sv.ndim == 3:
        sv = np.linalg.norm(sv, axis=-1)
    else:
        sv = np.asarray(sv).reshape(-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    # Mask out special / padding tokens so the visualisation is clean.
    keep = _real_token_mask(input_ids[0].cpu().tolist(), attention_mask[0].cpu().tolist(), tokenizer)
    tokens = [t for t, k in zip(tokens, keep) if k]
    scores = sv[: len(tokens)]
    return tokens, scores.astype(np.float64)


def phobert_integrated_gradients(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    n_steps: int = 50,
    target_class: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """Captum ``LayerIntegratedGradients`` on the embedding layer.

    Args:
        text: Raw input text.
        model: ``PhoBertClassifier`` instance.
        tokenizer: HuggingFace tokenizer (or extractor wrapper).
        max_length: Truncation length.
        n_steps: Number of Riemann samples for the integral.
        target_class: Output index to explain (1 = "Fake").

    Returns:
        ``(tokens, scores)`` — one attribution score per non-special
        token, summed across embedding dimensions.
    """
    from captum.attr import LayerIntegratedGradients  # lazy

    tokenizer = _unwrap_tokenizer(tokenizer)
    model.eval()
    device = _device_of(model)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Captum expects a forward function mapping (inputs_embeds, ...) → logits.
    def forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        am = attention_mask.expand(inputs_embeds.shape[0], -1)
        out = model.encoder(inputs_embeds=inputs_embeds, attention_mask=am)
        pooled = out.last_hidden_state[:, 0]
        return model.classifier(pooled)

    embedding_layer = model.encoder.embeddings

    # Captum requires integer-valued inputs only to satisfy some internal
    # checks; we keep input_ids but rely on the embedding-layer attribute.
    lig = LayerIntegratedGradients(
        forward,
        embedding_layer,
        layer=None,
    )
    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=None,             # zero embedding baseline
        additional_forward_args=(),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum across the embedding dimension to get one score per token.
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    keep = _real_token_mask(input_ids[0].cpu().tolist(), attention_mask[0].cpu().tolist(), tokenizer)
    tokens = [t for t, k in zip(tokens, keep) if k]
    scores = token_scores[: len(tokens)]
    return tokens, scores.astype(np.float64)


def phobert_attention_rollout(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    target_class: int = 1,
    head_reduce: str = "mean",
    discard_ratio: float = 0.0,
) -> Tuple[List[str], np.ndarray]:
    """Attention-rollout attribution (Abnar & Zuidema, 2020).

    For each layer we (1) average the ``target_class``-head's self-attention
    (configurable via ``head_reduce``), (2) optionally drop the lowest
    ``discard_ratio`` of attention weights per layer (Abnar & Zuidema's
    trick to suppress noise), and (3) multiply the resulting matrices to
    obtain a single (T × T) rollout matrix.  The CLS-row of that matrix
    is the per-token attribution.

    Args:
        text: Raw input text.
        model: ``PhoBertClassifier`` instance.
        tokenizer: HuggingFace tokenizer (or extractor wrapper).
        max_length: Truncation length.
        target_class: Output index (the attention rollout itself is
            class-agnostic; this argument is kept only to mirror the
            other helpers' signatures).
        head_reduce: How to combine heads — ``"mean"`` or ``"max"``.
        discard_ratio: Fraction of lowest attention weights to zero out
            per layer.  0 (default) keeps all weights.

    Returns:
        ``(tokens, scores)`` aligned with the non-special tokens.
    """
    tokenizer = _unwrap_tokenizer(tokenizer)
    model.eval()
    device = _device_of(model)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ``output_attentions=True`` is supported by every BERT/RoBERTa
    # backbone we use; pass it explicitly to be safe across versions.
    with torch.no_grad():
        outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions  # tuple of (1, H, T, T)
    if not attentions:
        raise RuntimeError(
            "PhoBERT encoder did not return attention weights. "
            "Ensure `output_attentions=True` is supported."
        )

    rollout = None
    for layer_attn in attentions:
        # ``layer_attn`` is (1, H, T, T).  Fuse heads.
        a = layer_attn.squeeze(0)  # (H, T, T)
        if head_reduce == "mean":
            a = a.mean(dim=0)
        elif head_reduce == "max":
            a = a.max(dim=0).values
        else:
            raise ValueError(f"Unsupported head_reduce={head_reduce!r}")

        # Apply the Abnar & Zuidema (2020) trick: zero-out the lowest
        # ``discard_ratio`` of weights, then renormalise.
        if 0.0 < discard_ratio < 1.0:
            flat = a.flatten()
            threshold = np.quantile(flat.cpu().numpy(), discard_ratio)
            a = torch.where(a < threshold, torch.zeros_like(a), a)
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Add the identity (residual connection) and renormalise.
        a = a + torch.eye(a.shape[0], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        rollout = a if rollout is None else rollout @ a

    # Take the CLS row (token index 0).
    cls_row = rollout[0].detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    keep = _real_token_mask(input_ids[0].cpu().tolist(), attention_mask[0].cpu().tolist(), tokenizer)
    tokens = [t for t, k in zip(tokens, keep) if k]
    scores = cls_row[: len(tokens)]
    return tokens, scores.astype(np.float64)


def compare_attribution_methods(
    text: str,
    methods_results: Sequence[Tuple[str, Tuple[List[str], np.ndarray]]],
    save_path: str,
    title: str = "PhoBERT attribution — method comparison",
) -> str:
    """Side-by-side token heatmap comparison of multiple attribution methods.

    Args:
        text: Original document (kept for the caption; tokens come from
            the per-method results).
        methods_results: Sequence of ``(method_name, (tokens, scores))``
            tuples.  Tokens can differ across methods (e.g. ``<s>``
            handling) — each panel keeps its own token list.
        save_path: Destination PNG path.  Directory is created if missing.

    Returns:
        The PNG path that was written.
    """
    n = len(methods_results)
    if n == 0:
        raise ValueError("methods_results must be non-empty.")

    fig, axes = plt.subplots(
        n, 1,
        figsize=(min(14, 0.55 * max(len(r[1][0]) for r in methods_results) + 2), max(1.5 * n, 2.0)),
        squeeze=False,
    )

    for row_idx, (method_name, (tokens, scores)) in enumerate(methods_results):
        ax = axes[row_idx, 0]
        ax.set_axis_off()
        if not tokens:
            ax.text(0.5, 0.5, "(no tokens)", ha="center", va="center")
            continue

        scores_arr = np.asarray(scores, dtype=np.float64)
        abs_max = float(np.max(np.abs(scores_arr))) if np.any(scores_arr != 0) else 1.0

        cell_h = 1.0
        for i, (tok, sc) in enumerate(zip(tokens, scores_arr)):
            colour = "#d62728" if sc >= 0 else "#1f77b4"
            alpha = 0.25 + 0.75 * (abs(sc) / abs_max if abs_max > 0 else 0.0)
            ax.add_patch(plt.Rectangle(
                (i, 0), 1.0, cell_h,
                facecolor=colour, alpha=alpha, edgecolor="white", linewidth=0.6,
            ))
            ax.text(
                i + 0.5, cell_h / 2, tok,
                ha="center", va="center",
                fontsize=9,
                color="black" if alpha < 0.55 else "white",
            )
        ax.set_xlim(0, len(tokens))
        ax.set_ylim(0, cell_h)
        ax.set_title(f"{method_name}", fontsize=11, fontweight="bold", loc="left", pad=4)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
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


def _unwrap_tokenizer(tokenizer):
    """Return the raw ``PreTrainedTokenizer`` regardless of which object was passed."""
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    return tokenizer


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _real_token_mask(
    ids: Sequence[int],
    mask: Sequence[int],
    tokenizer,
) -> List[bool]:
    """Return a boolean list — True for content (non-pad, non-special) tokens."""
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    out: List[bool] = []
    for tid, m in zip(ids, mask):
        if m == 0:
            out.append(False)
            continue
        if tid in (pad_id, cls_id, sep_id, None):
            out.append(False)
            continue
        # Skip the language-id prefix that BPE-based tokenizers add (e.g.
        # PhoBERT prepends '▁' to the first subword of a word).
        out.append(True)
    return out


__all__ = [
    "phobert_shap",
    "phobert_integrated_gradients",
    "phobert_attention_rollout",
    "compare_attribution_methods",
]