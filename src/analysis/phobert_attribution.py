from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def phobert_shap(
    text: str,
    model,
    tokenizer,
    max_length: int = 256,
    n_samples: int = 50,
    target_class: int = 1,
) -> Tuple[List[str], np.ndarray]:
    import shap

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

    embedding_layer = model.encoder.embeddings
    with torch.no_grad():
        baseline_embeds = torch.zeros_like(embedding_layer(input_ids))
        background_embeds = embedding_layer(input_ids).clone()

    def f(embeds: np.ndarray) -> np.ndarray:
        embeds_t = torch.as_tensor(embeds, dtype=baseline_embeds.dtype, device=device)
        if embeds_t.dim() == 2:
            embeds_t = embeds_t.unsqueeze(0)
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

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[target_class])
    else:
        sv = np.asarray(shap_values)[..., target_class]
    if sv.ndim == 3:
        sv = np.linalg.norm(sv, axis=-1)
    else:
        sv = np.asarray(sv).reshape(-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
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
    from captum.attr import LayerIntegratedGradients

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

    def forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        am = attention_mask.expand(inputs_embeds.shape[0], -1)
        out = model.encoder(inputs_embeds=inputs_embeds, attention_mask=am)
        pooled = out.last_hidden_state[:, 0]
        return model.classifier(pooled)

    embedding_layer = model.encoder.embeddings

    lig = LayerIntegratedGradients(
        forward,
        embedding_layer,
        layer=None,
    )
    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=None,
        additional_forward_args=(),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

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

    with torch.no_grad():
        outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions 
    if not attentions:
        raise RuntimeError(
            "PhoBERT encoder did not return attention weights. "
            "Ensure `output_attentions=True` is supported."
        )

    rollout = None
    for layer_attn in attentions:
        a = layer_attn.squeeze(0)
        if head_reduce == "mean":
            a = a.mean(dim=0)
        elif head_reduce == "max":
            a = a.max(dim=0).values
        else:
            raise ValueError(f"Unsupported head_reduce={head_reduce!r}")

        if 0.0 < discard_ratio < 1.0:
            flat = a.flatten()
            threshold = np.quantile(flat.cpu().numpy(), discard_ratio)
            a = torch.where(a < threshold, torch.zeros_like(a), a)
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        a = a + torch.eye(a.shape[0], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        rollout = a if rollout is None else rollout @ a

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


def _unwrap_tokenizer(tokenizer):
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
        out.append(True)
    return out


__all__ = [
    "phobert_shap",
    "phobert_integrated_gradients",
    "phobert_attention_rollout",
    "compare_attribution_methods",
]