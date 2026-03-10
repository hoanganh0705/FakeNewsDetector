"""Tests for src.preprocessing.text_preprocessor.TextPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor.clean_text()."""

    def setup_method(self):
        self.pp = TextPreprocessor()

    def test_removes_urls(self):
        text = "Đọc thêm tại https://example.com và www.foo.com hôm nay"
        result = self.pp.clean_text(text)
        assert "https://" not in result
        assert "www.foo.com" not in result

    def test_removes_html_tags(self):
        text = "<p>Tin tức <b>quan trọng</b></p>"
        result = self.pp.clean_text(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "tin tức" in result  # clean_text() lowercases input

    def test_removes_email(self):
        text = "Liên hệ qua email: contact@example.com để biết thêm."
        result = self.pp.clean_text(text)
        assert "contact@example.com" not in result

    def test_preserves_vietnamese_diacritics(self):
        text = "Chính phủ Việt Nam thông báo kế hoạch mới"
        result = self.pp.clean_text(text)
        assert "việt" in result  # lowercase preserved
        assert "chính" in result

    def test_empty_string_returns_empty(self):
        assert self.pp.clean_text("") == ""

    def test_non_string_returns_empty(self):
        assert self.pp.clean_text(None) == ""
        assert self.pp.clean_text(123) == ""

    def test_extra_whitespace_collapsed(self):
        text = "Tin   tức   hôm   nay"
        result = self.pp.clean_text(text)
        assert "  " not in result  # no double spaces

    def test_fit_transform_returns_matrix(self):
        """Vectorization now lives in TfidfFeatureExtractor, not TextPreprocessor."""
        from src.features.tfidf_features import TfidfFeatureExtractor

        texts = pd.Series(["tin tức thật", "tin giả không đúng", "bài báo quan trọng"])
        extractor = TfidfFeatureExtractor(max_features=100, min_df=1)
        X = extractor.fit_transform(texts)
        assert X.shape[0] == 3  # 3 rows
        assert X.shape[1] > 0  # some features
