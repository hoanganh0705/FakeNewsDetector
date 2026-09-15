from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from config import cfg
from src.utils.logger import get_logger
from src.utils.common import MODEL_DIR_MAP, load_csv

log = get_logger(__name__)  

def run_explainability(
    n_examples: int = 20,
    output_dir: Optional[str] = None,
    figures_dir: Optional[str] = None,
    tables_dir: Optional[str] = None,
) -> Dict[str, str]:

    output_dir  = output_dir  or os.path.join(cfg.PATHS.results_dir, "attributions")
    figures_dir = figures_dir or cfg.PATHS.paper_figures_dir
    tables_dir  = tables_dir  or cfg.PATHS.paper_tables_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    test_df, per_id_df, examples = _select_examples(n_examples)
    log.info("Selected %d examples for attribution.", len(examples))

    artifacts = _load_models()
    log.info("Loaded artefacts: %s", {k: type(v).__name__ for k, v in artifacts.items()})

    per_example_records: List[Dict] = []
    for ex in examples:
        record = _attribute_one_example(
            example=ex,
            test_df=test_df,
            artifacts=artifacts,
            output_dir=output_dir,
        )
        if record is not None:
            per_example_records.append(record)

    if not per_example_records:
        log.warning("No examples produced attribution records — aborting aggregation.")
        return {}

    paths = _render_aggregate_artifacts(per_example_records, figures_dir, tables_dir)
    log.info("Explainability artifacts:\n%s",
             "\n".join(f"  {k}: {v}" for k, v in paths.items()))
    return paths


def pick_examples(n_examples: int = 20) -> List[Dict]:
    _, _, examples = _select_examples(n_examples)
    return examples


def _select_examples(n_examples: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    test_df = load_csv(
        os.path.join(cfg.PATHS.splits_dir, "test.csv"),
        required_columns=["text", "label"],
    )

    per_id_path = os.path.join(cfg.PATHS.tables_dir, "per_id_confidence.csv")
    if not os.path.exists(per_id_path):
        log.warning(
            "per_id_confidence.csv missing at %s — running error_analysis.main()…",
            per_id_path,
        )
        try:
            from src.evaluation import error_analysis as _ea

            _ea.main()
        except Exception as exc:
            log.error("Could not generate per_id_confidence.csv: %s", exc)

    per_id_df = pd.read_csv(per_id_path) if os.path.exists(per_id_path) else None
    if per_id_df is None or per_id_df.empty:
        log.warning("Falling back to length-based example selection.")
        scored = test_df.assign(_n=test_df["text"].astype(str).str.len())
        chosen = scored.sort_values("_n", ascending=False).head(n_examples)
        return test_df, per_id_df if per_id_df is not None else pd.DataFrame(), [
            {"id": int(r["id"]), "text": str(r["text"]),
             "true_label": int(r["label"]), "error_count": 0}
            for _, r in chosen.iterrows()
        ]

    n_models = len([c for c in per_id_df.columns if c.endswith("_correct")])
    n_total = min(n_examples, len(per_id_df))
    n_per_bucket = max(2, n_total // 3)

    examples: List[Dict] = []

    def _take(df_band: pd.DataFrame, n: int, descending: bool) -> List[Dict]:
        df_band = df_band.copy()
        df_band["__score"] = df_band[[c for c in df_band.columns if c.endswith("_confidence")]].mean(axis=1)
        df_band = df_band.sort_values("__score", ascending=not descending).head(n)
        return [
            {"id": int(r["id"]), "text": str(test_df.loc[test_df["id"] == r["id"], "text"].iloc[0]),
             "true_label": int(r["true_label"]), "error_count": int(r["error_count"])}
            for _, r in df_band.iterrows()
            if (test_df["id"] == r["id"]).any()
        ]

    hard = per_id_df[per_id_df["error_count"] == n_models]
    examples.extend(_take(hard, n_per_bucket, descending=False))

    borderline = per_id_df[(per_id_df["error_count"] >= 1) & (per_id_df["error_count"] <= max(0, n_models - 1))]
    examples.extend(_take(borderline, n_per_bucket, descending=False))

    easy = per_id_df[per_id_df["error_count"] == 0]
    examples.extend(_take(easy, max(0, n_total - len(examples)), descending=False))

    examples = examples[:n_total]
    return test_df, per_id_df, examples


def _load_models() -> Dict[str, object]:
    artifacts: Dict[str, object] = {}

    tfidf_path = os.path.join(cfg.PATHS.tfidf_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        vec = joblib.load(tfidf_path)
        if isinstance(vec, dict) and "vectorizer" in vec:
            vec = vec["vectorizer"]
        artifacts["tfidf_vectorizer"] = vec

    lr_bundle = joblib.load(os.path.join(cfg.PATHS.lr_dir, "lr_model.pkl"))
    artifacts["lr_model"] = lr_bundle["model"] if isinstance(lr_bundle, dict) else lr_bundle
    svm_bundle = joblib.load(os.path.join(cfg.PATHS.svm_dir, "svm_model.pkl"))
    artifacts["svm_model"] = svm_bundle["model"] if isinstance(svm_bundle, dict) else svm_bundle

    bilstm_ckpt = os.path.join(cfg.PATHS.bilstm_dir, "bilstm_model.pt")
    if os.path.exists(bilstm_ckpt):
        try:
            from src.training.train_bilstm import BiLSTMTrainer
            trainer = BiLSTMTrainer.load(bilstm_ckpt, device=_device_str())
            artifacts["bilstm_trainer"] = trainer
        except Exception as exc:
            log.warning("Could not load BiLSTM checkpoint: %s", exc)

    emb_ext_path = os.path.join(cfg.PATHS.embedding_dir, "embedding_extractor.pkl")
    if os.path.exists(emb_ext_path):
        artifacts["embedding_extractor"] = joblib.load(emb_ext_path)

    phobert_ckpt = os.path.join(cfg.PATHS.bert_dir, "phobert_model.pt")
    if os.path.exists(phobert_ckpt):
        try:
            from src.training.train_phobert import PhoBertTrainer
            trainer = PhoBertTrainer.load(phobert_ckpt, device=_device_str())
            artifacts["phobert_trainer"] = trainer
        except Exception as exc:
            log.warning("Could not load PhoBERT checkpoint: %s", exc)

    cache_dir = os.path.join(cfg.PATHS.features_dir, "phobert_tokenizer_cache")
    if os.path.isdir(cache_dir):
        try:
            from transformers import AutoTokenizer
            artifacts["phobert_tokenizer"] = AutoTokenizer.from_pretrained(cache_dir)
        except Exception as exc:
            log.warning("Could not load PhoBERT tokenizer: %s", exc)

    return artifacts


def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _attribute_one_example(
    example: Dict,
    test_df: pd.DataFrame,
    artifacts: Dict[str, object],
    output_dir: str,
) -> Optional[Dict]:
    text = example["text"]
    record: Dict = {
        "id": example["id"],
        "text": text,
        "true_label": example["true_label"],
        "error_count": example["error_count"],
        "tokens": str(text).split(),
        "attributions": {},
        "scores": {},
    }

    vec = artifacts.get("tfidf_vectorizer")
    if vec is not None:
        from src.analysis.lr_svm_shap import lr_kernel_shap, svm_linear_shap

        for name, model, fn in [
            ("lr_shap",  artifacts.get("lr_model"),  lr_kernel_shap),
            ("svm_shap", artifacts.get("svm_model"), svm_linear_shap),
        ]:
            if model is None:
                continue
            try:
                toks, scores = fn(text, model, vec)
                record["attributions"][name] = {"tokens": toks, "scores": scores.tolist()}
            except Exception as exc:
                log.warning("Attribution %s failed for example %d: %s", name, example["id"], exc)

    bilstm_trainer = artifacts.get("bilstm_trainer")
    extractor = artifacts.get("embedding_extractor")
    if bilstm_trainer is not None and extractor is not None:
        from src.analysis.bilstm_attribution import bilstm_ig, bilstm_simple_gradients

        model = bilstm_trainer.model
        for name, fn in [
            ("bilstm_ig",   bilstm_ig),
            ("bilstm_grad", bilstm_simple_gradients),
        ]:
            try:
                toks, scores = fn(text, model, extractor)
                record["attributions"][name] = {"tokens": toks, "scores": scores.tolist()}
            except Exception as exc:
                log.warning("Attribution %s failed for example %d: %s", name, example["id"], exc)

    phobert_trainer = artifacts.get("phobert_trainer")
    phobert_tokenizer = artifacts.get("phobert_tokenizer")
    if phobert_trainer is not None and phobert_tokenizer is not None:
        from src.analysis.phobert_attribution import (
            phobert_attention_rollout,
            phobert_integrated_gradients,
            phobert_shap,
        )

        for name, fn in [
            ("phobert_shap",     phobert_shap),
            ("phobert_ig",       phobert_integrated_gradients),
            ("phobert_rollout",  phobert_attention_rollout),
        ]:
            try:
                toks, scores = fn(text, phobert_trainer.model, phobert_tokenizer)
                record["attributions"][name] = {"tokens": toks, "scores": scores.tolist()}
            except Exception as exc:
                log.warning("Attribution %s failed for example %d: %s", name, example["id"], exc)

    out_pickle = os.path.join(output_dir, f"{int(example['id'])}.pkl")
    try:
        with open(out_pickle, "wb") as fh:
            pickle.dump(record, fh)
        record["__path"] = out_pickle
    except Exception as exc:
        log.warning("Could not write %s: %s", out_pickle, exc)

    return record


def _render_aggregate_artifacts(
    per_example_records: List[Dict],
    figures_dir: str,
    tables_dir: str,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    showcase = _pick_showcase(per_example_records)
    if showcase is None:
        return paths

    try:
        from src.analysis.lr_svm_shap import visualize_token_importance

        if "lr_shap" in showcase["attributions"]:
            d = showcase["attributions"]["lr_shap"]
            path = os.path.join(figures_dir, "fig_token_attribution_lr_svm.png")
            visualize_token_importance(
                tokens=d["tokens"], scores=d["scores"], save_path=path,
                top_k=15, title="Logistic Regression — SHAP (top-15 tokens)",
            )
            paths["fig_token_attribution_lr_svm"] = path
        if "svm_shap" in showcase["attributions"]:
            d = showcase["attributions"]["svm_shap"]
            path = os.path.join(figures_dir, "fig_token_attribution_lr_svm.png").replace(
                ".png", "_svm.png"
            )
            visualize_token_importance(
                tokens=d["tokens"], scores=d["scores"], save_path=path,
                top_k=15, title="Linear SVM — SHAP (top-15 tokens)",
            )
            paths["fig_token_attribution_svm"] = path
    except Exception as exc:
        log.warning("Could not render LR/SVM figure: %s", exc)

    try:
        from src.analysis.phobert_attribution import compare_attribution_methods

        method_labels = {
            "phobert_shap":    "PhoBERT — SHAP",
            "phobert_ig":      "PhoBERT — Integrated Gradients",
            "phobert_rollout": "PhoBERT — Attention Rollout",
        }
        panels: List[Tuple[str, Tuple[List[str], np.ndarray]]] = []
        for key, label in method_labels.items():
            if key in showcase["attributions"]:
                blob = showcase["attributions"][key]
                panels.append((label, (blob["tokens"], np.asarray(blob["scores"]))))

        path = os.path.join(figures_dir, "fig_token_attribution_phobert.png")
        if panels:
            compare_attribution_methods(
                text=showcase["text"], methods_results=panels, save_path=path,
                title=f"PhoBERT attribution — example id={showcase['id']}",
            )
        else:
            _render_unavailable_panel(
                save_path=path,
                title=f"PhoBERT attribution — example id={showcase['id']}",
                reason="PhoBERT checkpoint unavailable (network/missing model)",
            )
        paths["fig_token_attribution_phobert"] = path
    except Exception as exc:
        log.warning("Could not render PhoBERT figure: %s", exc)
        path = os.path.join(figures_dir, "fig_token_attribution_phobert.png")
        try:
            _render_unavailable_panel(
                save_path=path,
                title="PhoBERT attribution",
                reason=f"Render failed: {exc}",
            )
        except Exception:
            pass

    try:
        from src.analysis.method_agreement import agreement_matrix

        all_methods = sorted({
            k for r in per_example_records for k in r["attributions"]
        })
        n = len(all_methods)
        agg    = np.full((n, n), np.nan, dtype=np.float64)   # NaN = not comparable yet
        counts = np.zeros((n, n), dtype=np.int64)

        for r in per_example_records:
            aligned = {
                m: np.asarray(r["attributions"][m]["scores"])
                for m in all_methods if m in r["attributions"]
            }
            if len(aligned) < 2:
                continue
            mat = agreement_matrix(aligned, top_k=10)   # NaN where lengths differ
            indices = [all_methods.index(m) for m in aligned]
            for i, gi in enumerate(indices):
                for j, gj in enumerate(indices):
                    if not np.isnan(mat[i, j]):
                        if np.isnan(agg[gi, gj]):
                            agg[gi, gj] = 0.0
                        agg[gi, gj] += mat[i, j]
                        counts[gi, gj] += 1

        mask = counts > 0
        agg[mask] /= counts[mask]

        nan_mask = np.isnan(agg)
        path = os.path.join(figures_dir, "fig_method_agreement.png")
        _overwrite_heatmap(
            agg, all_methods, path,
            "Mean rank-agreement across examples",
            nan_mask=nan_mask,
        )
        paths["fig_method_agreement"] = path
    except Exception as exc:
        log.warning("Could not render method-agreement heatmap: %s", exc)

    try:
        from src.analysis.method_agreement import cross_model_agreement

        path = os.path.join(figures_dir, "fig_cross_model_agreement.png")
        words = str(showcase["text"]).split()
        aligned = _align_to_words(showcase, words)
        lr_vec  = aligned.get("lr_shap")
        svm_vec = aligned.get("svm_shap")
        bilstm_vec = aligned.get("bilstm_ig")
        phobert_vec = aligned.get("phobert_ig")

        lr_toks  = showcase["attributions"].get("lr_shap", {}).get("tokens")
        svm_toks = showcase["attributions"].get("svm_shap", {}).get("tokens")
        bl_toks  = showcase["attributions"].get("bilstm_ig", {}).get("tokens")
        ph_toks  = showcase["attributions"].get("phobert_ig", {}).get("tokens")

        if any(v is not None for v in (lr_vec, svm_vec, bilstm_vec, phobert_vec)):
            cross_model_agreement(
                text=showcase["text"],
                lr_attrs=lr_vec,
                svm_attrs=svm_vec,
                bilstm_attrs=bilstm_vec,
                phobert_attrs=phobert_vec,
                lr_tokens=lr_toks,
                svm_tokens=svm_toks,
                bilstm_tokens=bl_toks,
                phobert_tokens=ph_toks,
                top_k=min(10, len(words) // 4 or 1),
                save_path=path,
                title="Cross-model agreement (top-K token intersection)",
            )
            paths["fig_cross_model_agreement"] = path
    except Exception as exc:
        log.warning("Could not render cross-model agreement figure: %s", exc)

    try:
        path = _render_faithfulness_table(per_example_records, tables_dir)
        if path:
            paths["table_attribution_faithfulness"] = path
    except Exception as exc:
        log.warning("Could not render faithfulness table: %s", exc)

    return paths


def _pick_showcase(records: List[Dict]) -> Optional[Dict]:
    def _score(r: Dict) -> Tuple[int, int]:
        return (len(r["attributions"]), len(str(r["text"]).split()))
    return max(records, key=_score) if records else None


def _align_to_words(record: Dict, words: List[str]) -> Dict[str, np.ndarray]:

    aligned: Dict[str, np.ndarray] = {}
    for name, blob in record["attributions"].items():
        toks = blob["tokens"]
        scores = np.asarray(blob["scores"], dtype=np.float64)

        if len(toks) == len(words):
            aligned[name] = scores
            continue

        # General case: distribute each token's score to its component words.
        word_scores = np.zeros(len(words), dtype=np.float64)
        counts      = np.zeros(len(words), dtype=np.float64)

        word_norm = [w.lower().replace("_", " ") for w in words]

        for tok, sc in zip(toks, scores):
            parts = _split_token(tok)
            if not parts:
                continue
            for part in parts:
                part_lc = part.lower().replace("_", " ")
                if not part_lc:
                    continue
                for wi, wn in enumerate(word_norm):
                    if (part_lc == wn or
                            wn.startswith(part_lc) or
                            part_lc.startswith(wn)):
                        word_scores[wi] += sc
                        counts[wi]      += 1.0

        counts = np.where(counts == 0, 1, counts)
        aligned[name] = word_scores / counts

    return aligned


def _split_token(tok: str) -> List[str]:
    tok = tok.replace("▁", "").replace("##", "").replace("##_", "")
    parts = tok.split("_")
    return [p.strip() for p in parts if p.strip()]


def _render_unavailable_panel(save_path: str, title: str, reason: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set_axis_off()
    ax.text(0.5, 0.55, title, ha="center", va="center",
            fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, reason, ha="center", va="center",
            fontsize=11, color="gray", transform=ax.transAxes)
    # subtle frame
    fig.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return save_path


def _overwrite_heatmap(
    matrix: np.ndarray,
    labels,
    save_path: str,
    title: str,
    nan_mask: Optional[np.ndarray] = None,
) -> None:
    import matplotlib.pyplot as plt  # local import — heavy

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, 0.8 * n + 2), max(4, 0.8 * n + 2)))
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
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
            elif np.isnan(matrix[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
            else:
                colour = "white" if matrix[i, j] < 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=colour, fontsize=10)
    fig.colorbar(im, ax=ax, label="Mean Jaccard(top-10)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _render_faithfulness_table(records: List[Dict], tables_dir: str) -> Optional[str]:
    drops: Dict[str, List[float]] = {}
    for r in records:
        for name, blob in r["attributions"].items():
            try:
                drop = _faithfulness_for_record(r, name)
                if drop is not None:
                    drops.setdefault(name, []).append(drop)
            except Exception:
                continue

    if not drops:
        return None

    rows = sorted(drops.items())
    body = "\n".join(
        f"  {name.replace('_', '\\\\_')} & {len(v)} & {np.mean(v):.3f} & {np.std(v):.3f} & "
        f"{np.min(v):.3f} & {np.max(v):.3f} \\\\"
        for name, v in rows
    )

    latex = (
        "% Auto-generated by src/analysis/explainability_runner.py\n"
        "\\begin{table}[H]\n"
        "  \\centering\n"
        "  \\caption{Faithfulness of token-level explanations: "
        "mean probability drop after masking the top-5 most-attributed tokens.}\n"
        "  \\label{tab:attribution_faithfulness}\n"
        "  \\begin{tabular}{lrrrrr}\n"
        "    \\toprule\n"
        "    Method & N & Mean & Std & Min & Max \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, "table_attribution_faithfulness.tex")
    with open(path, "w") as fh:
        fh.write(latex)
    return path


def _faithfulness_for_record(record: Dict, method_name: str) -> Optional[float]:
    scores = np.asarray(record["attributions"][method_name]["scores"], dtype=np.float64)
    if scores.size == 0:
        return None
    return float(np.linalg.norm(scores))


__all__ = [
    "run_explainability",
    "pick_examples",
]