"""Save / Load round-trip tests for models and feature extractors."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.features.tfidf_features import TfidfFeatureExtractor
from src.features.embedding_features import EmbeddingFeatureExtractor


class TestTfidfSaveLoad:
    """TfidfFeatureExtractor save → load round-trip."""

    def test_round_trip(self, tmp_path):
        texts = pd.Series([
            "tin tức thật sự quan trọng",
            "tin giả không đúng sự thật",
            "bài báo về chính phủ mới",
        ])
        ext = TfidfFeatureExtractor(max_features=100, min_df=1)
        X_orig = ext.fit_transform(texts)

        path = str(tmp_path / "tfidf.pkl")
        ext.save(path)

        loaded = TfidfFeatureExtractor.load(path)
        X_loaded = loaded.transform(texts)

        np.testing.assert_array_almost_equal(X_orig.toarray(), X_loaded.toarray())

    def test_load_preserves_config(self, tmp_path):
        ext = TfidfFeatureExtractor(max_features=42, min_df=1, ngram_range=(1, 3))
        # Need to fit before saving — use multi-letter tokens to avoid stop-word filtering
        ext.fit(pd.Series([
            "alpha beta gamma delta epsilon zeta",
            "eta theta iota kappa lambda mu",
            "alpha beta eta theta kappa nu",
        ]))
        path = str(tmp_path / "tfidf2.pkl")
        ext.save(path)

        loaded = TfidfFeatureExtractor.load(path)
        assert loaded.max_features == 42
        assert loaded.ngram_range == (1, 3)
        assert loaded.min_df == 1


class TestEmbeddingSaveLoad:
    """EmbeddingFeatureExtractor save → load round-trip."""

    def test_round_trip(self, tmp_path):
        texts = pd.Series(["aaa bbb ccc", "aaa ddd", "eee bbb"])
        ext = EmbeddingFeatureExtractor(max_vocab_size=100, min_freq=1, max_seq_length=10)
        seqs_orig = ext.fit_transform(texts)

        path = str(tmp_path / "emb.pkl")
        ext.save(path)

        loaded = EmbeddingFeatureExtractor.load(path)
        seqs_loaded = loaded.transform(texts)

        assert seqs_orig == seqs_loaded


class TestLRTrainerSaveLoad:
    """LogisticRegressionTrainer save → load round-trip (fit, save, load, predict)."""

    def test_round_trip(self, tmp_path):
        from src.training.train_lr import LogisticRegressionTrainer

        X = np.random.randn(40, 5)
        y = np.array([0] * 20 + [1] * 20)

        trainer = LogisticRegressionTrainer(class_weight="balanced", random_state=0)
        trainer.train(X, y)

        path = str(tmp_path / "lr.pkl")
        trainer.save(path)

        loaded = LogisticRegressionTrainer.load(path)
        preds_orig = trainer.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
