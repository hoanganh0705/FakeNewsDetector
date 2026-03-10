"""Tests for feature extractors (TF-IDF, Embedding, PhoBERT tokenizer)."""

import numpy as np
import pandas as pd
import pytest

from src.features.tfidf_features import TfidfFeatureExtractor
from src.features.embedding_features import EmbeddingFeatureExtractor, Vocabulary


# ──────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────

class TestVocabulary:

    def test_build_creates_word2idx(self):
        texts = pd.Series(["aaa bbb ccc", "aaa bbb ddd", "aaa eee fff"])
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(texts)
        assert "aaa" in vocab.word2idx
        assert vocab.PAD_TOKEN in vocab.word2idx
        assert vocab.UNK_TOKEN in vocab.word2idx

    def test_text_to_indices_returns_ints(self):
        texts = pd.Series(["aaa bbb", "aaa ccc"])
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(texts)
        indices = vocab.text_to_indices("aaa bbb")
        assert all(isinstance(i, (int, np.integer)) for i in indices)

    def test_unknown_word_maps_to_unk(self):
        texts = pd.Series(["hello world"])
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(texts)
        indices = vocab.text_to_indices("hello unknown_xyz_token")
        assert vocab.UNK_IDX in indices


# ──────────────────────────────────────────────────────────────
# TF-IDF
# ──────────────────────────────────────────────────────────────

class TestTfidfFeatureExtractor:

    @pytest.fixture()
    def sample_texts(self):
        return pd.Series([
            "tin tức thật sự quan trọng",
            "tin giả không đúng sự thật",
            "bài báo về chính phủ mới",
            "thông tin quan trọng từ nguồn chính phủ",
        ])

    def test_fit_transform_shape(self, sample_texts):
        ext = TfidfFeatureExtractor(max_features=50, min_df=1)
        X = ext.fit_transform(sample_texts)
        assert X.shape[0] == len(sample_texts)
        assert 0 < X.shape[1] <= 50

    def test_transform_after_fit(self, sample_texts):
        ext = TfidfFeatureExtractor(max_features=50, min_df=1)
        ext.fit(sample_texts)
        X = ext.transform(sample_texts[:2])
        assert X.shape[0] == 2

    def test_not_fitted_raises(self):
        ext = TfidfFeatureExtractor(max_features=50, min_df=1)
        with pytest.raises(ValueError, match="[Ff]it"):
            ext.transform(pd.Series(["foo bar"]))

    def test_get_feature_names(self, sample_texts):
        ext = TfidfFeatureExtractor(max_features=50, min_df=1)
        ext.fit_transform(sample_texts)
        names = ext.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0


# ──────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────

class TestEmbeddingFeatureExtractor:

    @pytest.fixture()
    def sample_texts(self):
        return pd.Series([
            "aaa bbb ccc ddd",
            "aaa bbb eee",
            "fff ggg aaa",
        ])

    def test_fit_transform_returns_list_of_lists(self, sample_texts):
        ext = EmbeddingFeatureExtractor(max_vocab_size=100, min_freq=1)
        seqs = ext.fit_transform(sample_texts)
        assert isinstance(seqs, list)
        assert all(isinstance(s, list) for s in seqs)

    def test_sequence_max_length(self, sample_texts):
        ext = EmbeddingFeatureExtractor(max_vocab_size=100, max_seq_length=5, min_freq=1)
        seqs = ext.fit_transform(sample_texts)
        for s in seqs:
            assert len(s) <= 5

    def test_vocab_size(self, sample_texts):
        ext = EmbeddingFeatureExtractor(max_vocab_size=100, min_freq=1)
        ext.fit(sample_texts)
        assert ext.vocab_size > 2  # at least PAD + UNK + some words


# ──────────────────────────────────────────────────────────────
# PhoBERT tokenizer — skipped if weights unavailable
# ──────────────────────────────────────────────────────────────

class TestPhoBertFeatureExtractor:

    @pytest.fixture(autouse=True)
    def _load_tokenizer(self):
        try:
            from src.features.phobert_features import PhoBertFeatureExtractor
            self.ext = PhoBertFeatureExtractor(max_length=32)
        except Exception:
            pytest.skip("PhoBERT tokenizer unavailable — skipping")

    def test_tokenize_returns_dict_with_expected_keys(self):
        result = self.ext.tokenize(pd.Series(["xin chào thế giới"]))
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tokenize_shape(self):
        result = self.ext.tokenize(pd.Series(["aaa bbb", "ccc ddd"]))
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == 32  # max_length
