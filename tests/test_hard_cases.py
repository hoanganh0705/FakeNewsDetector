"""Tests for ``src.evaluation.hard_cases`` (Phase 4).

These tests cover:
* The annotation pipeline (topic detection, reason categorisation, length bucket).
* ``annotate_one`` produces a ``HardExampleAnnotation`` with all fields.
* ``analyze_hard_examples`` returns a non-empty DataFrame and saves artefacts.
* The LaTeX / figure writers don't crash on empty or populated data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.hard_cases import (
    HardExampleAnnotation,
    annotate_one,
    _detect_topic,
    _categorise_reason,
    _length_bucket,
    _clean,
    _tokenise,
)


# ──────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────

class TestLengthBucket:
    @pytest.mark.parametrize("n,expected", [
        (10, "short (<30)"),
        (29, "short (<30)"),
        (30, "medium (30-99)"),
        (50, "medium (30-99)"),
        (99, "medium (30-99)"),
        (100, "long (100-249)"),
        (200, "long (100-249)"),
        (250, "very_long (≥250)"),
        (500, "very_long (≥250)"),
    ])
    def test_bucket(self, n, expected):
        assert _length_bucket(n) == expected


class TestTopicDetection:
    @pytest.mark.parametrize("text,expected_topic", [
        ("Trung Quốc tấn công Đài Loan", "politics"),
        ("COVID 19 lây lan tại Hà Nội", "covid_health"),
        ("cổ phiếu tăng trưởng 20%", "finance_business"),
        ("tai nạn giao thông nghiêm trọng", "crime_accident"),
        ("ca sĩ phát hành album mới", "entertainment"),
        ("world cup 2022 real madrid", "sports"),
        ("trí tuệ nhân tạo chatbot gpt", "science_tech"),
        ("tổ chức từ thiện nhà thờ", "religion_society"),
        ("thi thpt quốc gia điểm chuẩn", "education"),
        ("this is some random text", "other"),
    ])
    def test_topics(self, text, expected_topic):
        assert _detect_topic(text.lower()) == expected_topic


class TestReasonCategorisation:
    @pytest.mark.parametrize("text,has_urls,has_numbers,has_caps,n_tokens,label,expected", [
        # URL + numbers → knowledge_verification
        ("visit http://example.com for 2023 update", True, True, False, 8, 1, "knowledge_verification"),
        # 4-digit year → knowledge_verification
        ("GDP 2023 reached 5%", False, True, False, 6, 0, "knowledge_verification"),
        # long + no caps → semantic (text does NOT contain topic keywords, >=80 tokens)
        ("The phenomenon described in recent reports involves multiple dimensions " * 8, False, False, False, 80, 1, "semantic_reasoning"),
        # short + ALL CAPS → stylistic
        ("BREAKING NEWS TODAY TRUMP", False, False, True, 6, 0, "stylistic"),
        # very short + no cues → dataset_ambiguity
        ("just a short statement", False, False, False, 5, 1, "dataset_ambiguity"),
    ])
    def test_reasons(self, text, has_urls, has_numbers, has_caps, n_tokens, label, expected):
        result = _categorise_reason(
            text_lower=text.lower(),
            raw_text=text,
            n_tokens=n_tokens,
            has_urls=has_urls,
            has_numbers=has_numbers,
            has_all_caps=has_caps,
            label=label,
        )
        assert result == expected


class TestAnnotateOne:
    def test_returns_annotation_with_all_fields(self):
        ann = annotate_one(
            row_id=123,
            text_seg="việt_nam có_dân_số 100 triệu người .",
            text_raw="Việt Nam có dân số 100 triệu người.",
            label=1,
            date_str="2023",
            confidences={
                "Logistic Regression": 0.75,
                "SVM": 0.82,
                "BiLSTM": 0.91,
                "PhoBERT": 0.88,
            },
            attributions={"lr_shap": [("việt_nam", 0.5), ("dân_số", 0.3)]},
        )
        assert isinstance(ann, HardExampleAnnotation)
        assert ann.id == 123
        assert ann.label == 1
        assert ann.contains_numbers is True
        assert ann.contains_urls is False
        assert ann.contains_all_caps is False
        assert ann.topic == "politics"
        assert ann.reason == "knowledge_verification"
        assert ann.n_tokens == 8
        assert ann.lr_conf == 0.75
        assert ann.phobert_conf == 0.88
        assert ann.top_tokens == {"lr_shap": [("việt_nam", 0.5), ("dân_số", 0.3)]}

    def test_url_flag(self):
        ann = annotate_one(
            row_id=1, text_seg="click < URL > now", text_raw="Click http://x.com now",
            label=0, date_str="2022",
            confidences={"Logistic Regression": 0.5, "SVM": 0.5, "BiLSTM": 0.5, "PhoBERT": 0.5},
        )
        assert ann.contains_urls is True

    def test_to_dict_is_json_safe(self):
        ann = annotate_one(
            row_id=1, text_seg="test", text_raw="test", label=0, date_str="2022",
            confidences={"Logistic Regression": 0.5, "SVM": 0.5, "BiLSTM": 0.5, "PhoBERT": 0.5},
        )
        d = ann.to_dict()
        # to_dict must not raise and must be JSON-serialisable (tuple → list).
        import json
        json.dumps(d)  # raises if not serialisable


# ──────────────────────────────────────────────────────────────────────
# Pipeline — end-to-end with a minimal mock CSV
# ──────────────────────────────────────────────────────────────────────

class TestAnalyzeHardExamples:
    def test_returns_dataframe(self):
        """Smoke test: the pipeline returns a non-empty DataFrame on real data."""
        from src.evaluation.hard_cases import analyze_hard_examples
        with tempfile.TemporaryDirectory() as tmp:
            df = analyze_hard_examples(
                per_id_confidence_path="results/tables/per_id_confidence.csv",
                test_csv_path="data/splits/test.csv",
                tables_dir=tmp,
                figures_dir=tmp,
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 85
        for col in ("id", "label", "topic", "reason", "length_bucket", "lexical_density"):
            assert col in df.columns, f"Missing column: {col}"

    def test_saves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            from src.evaluation.hard_cases import analyze_hard_examples
            df = analyze_hard_examples(
                per_id_confidence_path="results/tables/per_id_confidence.csv",
                test_csv_path="data/splits/test.csv",
                tables_dir=tmp,
                figures_dir=tmp,
            )
            assert os.path.exists(os.path.join(tmp, "table_hard_cases_annotated.tex"))

    def test_distribution_figure_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            from src.evaluation.hard_cases import analyze_hard_examples
            analyze_hard_examples(
                per_id_confidence_path="results/tables/per_id_confidence.csv",
                test_csv_path="data/splits/test.csv",
                tables_dir=tmp,
                figures_dir=tmp,
            )
            assert os.path.exists(os.path.join(tmp, "fig_hard_case_distribution.png"))
