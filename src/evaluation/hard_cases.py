"""
Hard cases deep analysis (Phase 4 of IMPLEMENTATION_PLAN.md).

For every test sample that **all four** models misclassify, this module
produces a structured annotation:

* ``topic``        — keyword-based category (politics, health, COVID, ...)
* ``length_bucket`` — text length in tokens → short / medium / long / very_long
* ``lexical_density`` — type-token ratio (unique / total tokens)
* ``contains_numbers``, ``contains_urls``, ``contains_named_entities``
  — boolean flags
* ``reason``        — categorised failure mode:
    * ``"semantic_reasoning"`` — needs world knowledge beyond lexical cues
    * ``"knowledge_verification"`` — contains verifiable claims that turn out false
    * ``"dataset_ambiguity"`` — text is genuinely hard to label even for humans
    * ``"stylistic"`` — short, factual-looking fake; or long, opinion-style real
    * ``"other"``
* ``top_attributions`` — for each model, the top-3 tokens from Phase 1 SHAP/IG

Outputs:

* ``results/tables/hard_examples_annotated.csv`` — machine-readable
* ``paper/tables/table_hard_cases_annotated.tex``  — paper-ready table
* ``paper/figures/fig_hard_case_distribution.png`` — distribution by topic/length/reason

The annotator is **deterministic and rule-based** so the outputs are
fully reproducible without LLM calls.  Where a human would normally
be needed (semantic reason assignment), we use carefully designed
heuristics documented in ``_categorise_reason``.
"""

from __future__ import annotations

import os
import re
import json
import joblib
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import cfg
from src.utils.common import MODEL_DIR_MAP
from src.utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Topic & reason vocabularies (Vietnamese + English keywords)
# ──────────────────────────────────────────────────────────────────────

# Topics are checked in priority order: most specific topics first.
# IMPORTANT: covid_health must come BEFORE politics because the text
# "COVID-19 lây lan tại Hà Nội" contains "hà nội" which would match
# the politics block.  By checking covid first we avoid shadowing.
TOPIC_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("covid_health", [
        "covid", "covid 19", "covid19", "corona", "virus", "viêm phổi",
        "tiêm chủng", "vắc xin", "vaccine", "cách ly", "phong tỏa",
        "dịch bệnh", "đại dịch", "sars", "omicron", "delta",
        "bệnh viện", "bác sĩ", "f0", "f1", "sức khỏe", "thuốc",
        "điều trị", "ca nhiễm", "ca mắc", "xét nghiệm", "phổi",
        "đường hô hấp", "h5n1", "h1n1", "ebola", "mers", "who", "cdc",
    ]),
    ("politics", [
        "trung quốc", "việt nam", "mỹ", "đảng", "chính phủ", "thủ tướng",
        "tổng thống", "bộ trưởng", "quốc hội", "bầu cử", "đại biểu",
        "trump", "obama", "biden", "putin",
        "biểu tình", "hiến pháp", "đối lập", "thanh niên", "nhà nước",
        # NOTE: "hà nội" intentionally NOT included here — COVID articles
        # about Hanoi should be categorised as covid_health, not politics.
        "china", "vietnam", "u.s.", "u.s", "washington", "hcm", "tp.hcm",
        "bộ công an", "công an", "cảnh sát",
    ]),
    ("finance_business", [
        "usd", "vnd", "đô la", "tỷ", "triệu", "ngân hàng", "lãi suất",
        "chứng khoán", "cổ phiếu", "vàng", "bitcoin", "tiền ảo", "forex",
        "gdp", "tăng trưởng", "lạm phát", "thuế", "doanh thu", "lợi nhuận",
        "thị trường", "tài chính", "đầu tư", "kinh tế", "vn-index",
        "shark", "doanh nhân", "startup",
    ]),
    ("crime_accident", [
        "tai nạn", "chết", "thiệt mạng", "bị giết", "bắt giữ", "khởi tố",
        "cướp", "lừa đảo", "đánh bạc", "ma túy", " heroin", "ma túy đá",
        "bạo lực", "hiếp dâm", "giết người", "tình nghi", "truy nã",
        "án mạng", "tử vong",
    ]),
    ("entertainment", [
        "ca sĩ", "diễn viên", "nghệ sĩ", "showbiz", "phim", "mv ",
        "music", "bài hát", "album", "concert", "hát", "idol", "fan",
        "ngôi sao", "hoa hậu", "siêu mẫu", "thời trang", "đám hỏi",
        "đám cưới", "sao việt", "victoria",
    ]),
    ("sports", [
        "bóng đá", "world cup", "euro", "champions league", "premier league",
        "messi", "ronaldo", "neymar", "v-league", "u23", "sea games",
        "olympic", "world cup", "võ thuật", "boxing", "tennis", "golf",
        "huy chương", "cầu thủ", "huấn luyện viên",
    ]),
    ("science_tech", [
        "vũ trụ", "khám phá", "phát minh", "khoa học", "công nghệ",
        "trí tuệ nhân tạo", "ai ", "robot", "điện thoại", "iphone",
        "samsung", "google", "facebook", "tiktok", "youtube", "elon musk",
        "tesla", "spacex", "nasa", "vệ tinh", "phóng",
    ]),
    ("religion_society", [
        "phật", "chúa", "thánh", "đạo", "tôn giáo", "nhà thờ", "chùa",
        "tín đồ", "linh mục", "giáo sĩ", "hồi giáo", "thần",
        "đức tin", "phép màu",
    ]),
    ("education", [
        "học sinh", "sinh viên", "giáo viên", "trường", "đại học", "thpt",
        "thi ", "điểm", "kỳ thi", "tốt nghiệp", "ngành", "khoa", "lớp",
        "tiến sĩ", "giáo sư", "tiến sỹ", "bằng cấp",
    ]),
]

DEFAULT_TOPIC = "other"


# ──────────────────────────────────────────────────────────────────────
# Annotation primitives
# ──────────────────────────────────────────────────────────────────────

_RE_URL = re.compile(r"http[s]?://\S+|www\.\S+|< ?URL ?>", re.IGNORECASE)
_RE_NUMBER = re.compile(r"\b\d+([.,]\d+)*\b")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _clean(text: str) -> str:
    """Strip HTML tags + collapse whitespace."""
    text = _RE_HTML_TAG.sub(" ", text or "")
    return _RE_MULTI_SPACE.sub(" ", text).strip()


def _tokenise(text: str) -> List[str]:
    return [t for t in (text or "").split() if t]


def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 30:
        return "short (<30)"
    if n_tokens < 100:
        return "medium (30-99)"
    if n_tokens < 250:
        return "long (100-249)"
    return "very_long (≥250)"


def _detect_topic(text_lower: str) -> str:
    for topic, keywords in TOPIC_KEYWORDS:
        for kw in keywords:
            if kw in text_lower:
                return topic
    return DEFAULT_TOPIC


def _contains_numbers(text: str) -> bool:
    return bool(_RE_NUMBER.search(text))


def _contains_urls(text: str) -> bool:
    return bool(_RE_URL.search(text))


def _contains_named_entities(text: str) -> bool:
    """Crude NE heuristic: any uppercase word of length ≥2 + any
    name-shaped Vietnamese syllable.  We err on the side of inclusion."""
    if re.search(r"\b[A-Z][a-z]{2,}\b", text):
        return True
    if re.search(r"\b[A-ZÂĂĐÊÔƠƯ][a-zâăđêôơưáàảãạằẳẵặằẩẫậếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]+", text):
        return True
    if re.search(r"\b[A-ZÂĂĐÊÔƠƯ]{3,}\b", text):
        return True
    return False


def _contains_all_caps(text: str) -> bool:
    return bool(re.search(r"\b[A-ZÂĂĐÊÔƠƯ]{3,}\b", text))


def _parse_year(date_str: str) -> Optional[int]:
    if not isinstance(date_str, str) or not date_str:
        return None
    for pat in (r"\b(20\d{2})\b", r"\b(19\d{2})\b"):
        m = re.search(pat, date_str)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _categorise_reason(
    text_lower: str,
    raw_text: str,
    n_tokens: int,
    has_urls: bool,
    has_numbers: bool,
    has_all_caps: bool,
    label: int,
) -> str:
    """Heuristic failure-mode categoriser."""
    # knowledge_verification: numbers + URL → verifiable claim
    if has_urls and has_numbers:
        return "knowledge_verification"
    if has_numbers and re.search(r"\b\d{3,}\b", text_lower):
        return "knowledge_verification"
    # semantic_reasoning: long, prose-heavy, no surface cues
    if n_tokens >= 80 and not has_all_caps:
        return "semantic_reasoning"
    # stylistic: short + all-caps → clickbait fake / mistitled
    if has_all_caps and n_tokens < 80:
        return "stylistic"
    # dataset_ambiguity: very short, no surface cues
    if n_tokens < 25:
        return "dataset_ambiguity"
    return "other"


# ──────────────────────────────────────────────────────────────────────
# Annotation container
# ──────────────────────────────────────────────────────────────────────


@dataclass
class HardExampleAnnotation:
    id: int
    label: int
    text_raw: str
    text_seg: str
    text_snippet: str
    n_tokens: int
    length_bucket: str
    lexical_density: float
    contains_numbers: bool
    contains_urls: bool
    contains_named_entities: bool
    contains_all_caps: bool
    topic: str
    reason: str
    year: Optional[int]
    lr_conf: float
    svm_conf: float
    bilstm_conf: float
    phobert_conf: float
    top_tokens: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top_tokens"] = {
            k: [[tok, float(score)] for tok, score in v]
            for k, v in self.top_tokens.items()
        }
        return d


# ──────────────────────────────────────────────────────────────────────
# Phase 4.1 — Annotation pipeline
# ──────────────────────────────────────────────────────────────────────


def annotate_one(
    row_id: int,
    text_seg: str,
    text_raw: str,
    label: int,
    date_str: Optional[str],
    confidences: Dict[str, float],
    attributions: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> HardExampleAnnotation:
    """Annotate a single hard example."""
    cleaned_raw = _clean(text_raw)
    cleaned_seg = _clean(text_seg)
    raw_lower = cleaned_raw.lower()

    tokens = _tokenise(cleaned_raw)
    n_tokens = len(tokens)
    unique = set(t.lower() for t in tokens)
    density = (len(unique) / n_tokens) if n_tokens > 0 else 0.0

    has_numbers = _contains_numbers(cleaned_raw)
    has_urls = _contains_urls(cleaned_raw)
    has_ne = _contains_named_entities(cleaned_raw)
    has_all_caps = _contains_all_caps(cleaned_raw)
    topic = _detect_topic(raw_lower)
    year = _parse_year(date_str or "")

    reason = _categorise_reason(
        text_lower=raw_lower,
        raw_text=cleaned_raw,
        n_tokens=n_tokens,
        has_urls=has_urls,
        has_numbers=has_numbers,
        has_all_caps=has_all_caps,
        label=label,
    )

    snippet = cleaned_raw[:250]
    if len(cleaned_raw) > 250:
        snippet += "..."

    return HardExampleAnnotation(
        id=int(row_id),
        label=int(label),
        text_raw=cleaned_raw,
        text_seg=cleaned_seg,
        text_snippet=snippet,
        n_tokens=n_tokens,
        length_bucket=_length_bucket(n_tokens),
        lexical_density=round(density, 4),
        contains_numbers=has_numbers,
        contains_urls=has_urls,
        contains_named_entities=has_ne,
        contains_all_caps=has_all_caps,
        topic=topic,
        reason=reason,
        year=year,
        lr_conf=confidences.get("Logistic Regression", float("nan")),
        svm_conf=confidences.get("SVM", float("nan")),
        bilstm_conf=confidences.get("BiLSTM", float("nan")),
        phobert_conf=confidences.get("PhoBERT", float("nan")),
        top_tokens=attributions or {},
    )


# ──────────────────────────────────────────────────────────────────────
# Phase 4.3 — Cross-reference with Phase 1 attributions
# ──────────────────────────────────────────────────────────────────────


def _load_topk_attribution(
    attribution_path: Path,
    k: int = 3,
) -> Dict[str, List[Tuple[str, float]]]:
    """Load top-k tokens per attribution method from a Phase 1 pickle."""
    methods_to_keep = (
        "lr_shap", "svm_shap", "bilstm_ig",
        "phobert_shap", "phobert_ig", "phobert_rollout",
    )
    out: Dict[str, List[Tuple[str, float]]] = {}
    if not attribution_path.exists():
        return out
    try:
        bundle = joblib.load(attribution_path)
    except Exception as exc:
        log.warning("Could not load attribution %s: %s", attribution_path, exc)
        return out

    attrs = bundle.get("attributions", {})
    for m in methods_to_keep:
        if m not in attrs:
            continue
        entry = attrs[m]
        tokens = entry.get("tokens", [])
        scores = entry.get("scores", [])
        if not tokens or not scores:
            continue
        scores_arr = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-np.abs(scores_arr), kind="stable")[:k]
        out[m] = [(str(tokens[i]), float(scores_arr[i])) for i in order]
    return out


# ──────────────────────────────────────────────────────────────────────
# Phase 4.2 — Main entry point
# ──────────────────────────────────────────────────────────────────────


def analyze_hard_examples(
    per_id_confidence_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    raw_csv_path: Optional[str] = None,
    attributions_dir: Optional[str] = None,
    tables_dir: Optional[str] = None,
    figures_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full Phase 4 pipeline.

    1. Load per-ID confidence, keep only samples wrong by all models.
    2. Annotate each example with topic / length / density / reason / confidence.
    3. Cross-reference with Phase 1 SHAP / IG attributions if available.
    4. Write CSV + LaTeX + figure.
    """
    per_id_confidence_path = per_id_confidence_path or os.path.join(
        cfg.PATHS.tables_dir, "per_id_confidence.csv",
    )
    test_csv_path = test_csv_path or os.path.join(cfg.PATHS.splits_dir, "test.csv")
    attributions_dir = attributions_dir or os.path.join(cfg.PATHS.results_dir, "attributions")
    tables_dir = tables_dir or cfg.PATHS.paper_tables_dir
    figures_dir = figures_dir or cfg.PATHS.paper_figures_dir
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("  HARD CASES DEEP ANALYSIS (Phase 4)")
    log.info("=" * 70)

    if not os.path.exists(per_id_confidence_path):
        log.error("Missing %s — run error_analysis.main() first.", per_id_confidence_path)
        return pd.DataFrame()

    pid = pd.read_csv(per_id_confidence_path)
    n_models = sum(1 for c in pid.columns if c.endswith("_pred"))
    hard_ids = pid[pid["error_count"] == n_models]["id"].astype(int).tolist()
    log.info("Found %d hard examples (wrong by all %d models)", len(hard_ids), n_models)

    # NOTE: test.csv ids and raw.csv ids do NOT align after the split.
    # We use test.csv text directly (segmented but fully readable).
    test_df = pd.read_csv(test_csv_path)
    test_lookup = test_df.set_index("id")[["text", "date"]].to_dict("index")

    annotations: List[HardExampleAnnotation] = []
    n_with_attributions = 0
    for _, row in pid[pid["error_count"] == n_models].iterrows():
        ex_id = int(row["id"])
        label = int(row["true_label"])
        ex = test_lookup.get(ex_id, {})
        seg_text = str(ex.get("text", ""))
        date_str = ex.get("date")

        confidences = {
            "Logistic Regression": float(row["Logistic Regression_confidence"]),
            "SVM":                  float(row["SVM_confidence"]),
            "BiLSTM":               float(row["BiLSTM_confidence"]),
            "PhoBERT":              float(row["PhoBERT_confidence"]),
        }

        attr_path = Path(attributions_dir) / f"{ex_id}.pkl"
        attr_topk = _load_topk_attribution(attr_path, k=3)
        if attr_topk:
            n_with_attributions += 1

        annotations.append(annotate_one(
            row_id=ex_id,
            text_seg=seg_text,
            text_raw=seg_text,
            label=label,
            date_str=date_str,
            confidences=confidences,
            attributions=attr_topk,
        ))

    log.info("Cross-referenced %d / %d hard examples with Phase 1 attributions",
             n_with_attributions, len(annotations))

    df = pd.DataFrame([a.to_dict() for a in annotations])

    # Flatten top_tokens for the CSV.
    if not df.empty and "top_tokens" in df.columns:
        for method in (
            "lr_shap", "svm_shap", "bilstm_ig",
            "phobert_shap", "phobert_ig", "phobert_rollout",
        ):
            col_name = f"top_{method}"
            df[col_name] = df["top_tokens"].apply(
                lambda d, m=method: ", ".join(f"{t}:{s:+.2f}"
                                              for t, s in d.get(m, []))
                                          if d.get(m) else "",
            )
        df = df.drop(columns=["top_tokens"])

    csv_path = os.path.join(cfg.PATHS.results_dir, "tables", "hard_examples_annotated.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info("Saved annotated hard examples CSV → %s", csv_path)

    latex_path = os.path.join(tables_dir, "table_hard_cases_annotated.tex")
    _render_hard_cases_table(df, latex_path)
    log.info("Saved LaTeX table → %s", latex_path)

    fig_path = os.path.join(figures_dir, "fig_hard_case_distribution.png")
    _render_distribution_grid(df, fig_path)
    log.info("Saved distribution figure → %s", fig_path)

    log.info("")
    log.info("─" * 70)
    log.info("  Hard-case summary (n=%d)", len(df))
    log.info("─" * 70)
    if not df.empty:
        log.info("  Topic distribution:")
        for topic, n in df["topic"].value_counts().items():
            log.info("    %-20s %4d  (%.1f%%)", topic, n, 100.0 * n / len(df))
        log.info("  Length bucket distribution:")
        for bucket, n in df["length_bucket"].value_counts().items():
            log.info("    %-20s %4d  (%.1f%%)", bucket, n, 100.0 * n / len(df))
        log.info("  Reason distribution:")
        for reason, n in df["reason"].value_counts().items():
            log.info("    %-20s %4d  (%.1f%%)", reason, n, 100.0 * n / len(df))

    return df


# ──────────────────────────────────────────────────────────────────────
# Phase 4.4 — LaTeX table + figure
# ──────────────────────────────────────────────────────────────────────


def _render_hard_cases_table(df: pd.DataFrame, save_path: str, n_rows: int = 12) -> None:
    """Write ``table_hard_cases_annotated.tex`` with up to ``n_rows`` examples."""
    if df.empty:
        log.warning("Empty DataFrame — skipping LaTeX table.")
        return
    rows = df.head(n_rows)

    lines = [
        "% Auto-generated by src/evaluation/hard_cases.py",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Một số ví dụ điển hình mà cả bốn mô hình đều dự đoán sai. "
        r"Nhãn thật: 0 = Thật, 1 = Giả. Bảng thể hiện đặc trưng bề mặt (độ dài, "
        r"từ khóa, thực thể) cùng với chủ đề và lý do thất bại suy đoán.}",
        r"  \label{tab:hard-cases}",
        r"  \footnotesize",
        r"  \begin{tabular}{r r l l l l}",
        r"    \toprule",
        r"    \textbf{ID} & \textbf{Nhãn} & \textbf{Trích đoạn} & "
        r"\textbf{Chủ đề} & \textbf{Lý do} & \textbf{Độ dài} \\",
        r"    \midrule",
    ]
    for _, row in rows.iterrows():
        label = int(row["label"])
        label_tex = r"\textcolor{red}{\textbf{Giả}}" if label == 1 else r"\textbf{Thật}"
        snippet = str(row["text_snippet"]).replace("&", r"\&").replace("_", r"\_").replace("$", r"\$")
        if len(snippet) > 110:
            snippet = snippet[:107] + r"\ldots"
        topic = str(row["topic"]).replace("_", r"\_")
        reason = str(row["reason"]).replace("_", r"\_")
        bucket = str(row["length_bucket"]).replace("_", r"\_")
        lines.append(
            f"    {int(row['id'])} & {label_tex} & {snippet} & {topic} & {reason} & {bucket} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _render_distribution_grid(df: pd.DataFrame, save_path: str) -> None:
    """``fig_hard_case_distribution.png`` — 2 × 2 grid."""
    if df.empty:
        log.warning("Empty DataFrame — skipping distribution figure.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    counts = df["topic"].value_counts().sort_values(ascending=True)
    ax.barh(counts.index.astype(str), counts.values, color="#1f77b4")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("Số mẫu")
    ax.set_title("Phân bố chủ đề", fontweight="bold")

    ax = axes[0, 1]
    bucket_order = ["short (<30)", "medium (30-99)", "long (100-249)", "very_long (≥250)"]
    counts = df["length_bucket"].value_counts().reindex(bucket_order).fillna(0).astype(int)
    ax.bar(counts.index, counts.values, color="#ff7f0e")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân bố độ dài (từ)", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    counts = df["reason"].value_counts().sort_values(ascending=True)
    ax.barh(counts.index.astype(str), counts.values, color="#2ca02c")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("Số mẫu")
    ax.set_title("Phân bố lý do thất bại (reason)", fontweight="bold")

    ax = axes[1, 1]
    conf_cols = ["lr_conf", "svm_conf", "bilstm_conf", "phobert_conf"]
    means = [df[c].mean() for c in conf_cols]
    ax.bar(["LR", "SVM", "BiLSTM", "PhoBERT"], means,
           color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for i, v in enumerate(means):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Độ tự tin trung bình")
    ax.set_title("Độ tự tin trung bình trên hard examples", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Phân tích phân bố của các mẫu \"hard cases\" (cả 4 mô hình sai)",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — equivalent to ``fakenews hard-cases``."""
    analyze_hard_examples()


__all__ = [
    "HardExampleAnnotation",
    "annotate_one",
    "analyze_hard_examples",
    "main",
]
