"""
Integration test: synthetic data → features → train → evaluate.

Uses a small synthetic dataset to exercise the TF-IDF + Logistic Regression
pipeline end-to-end without requiring real data or GPU.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics
from src.features.tfidf_features import TfidfFeatureExtractor
from src.training.train_lr import LogisticRegressionTrainer


@pytest.fixture()
def synthetic_dataset():
    """Create a small synthetic Vietnamese-like text dataset."""
    rng = np.random.RandomState(42)
    n = 100
    real_words = ["chính phủ", "thông báo", "quan trọng", "kinh tế", "phát triển"]
    fake_words = ["giả mạo", "sốc", "bí mật", "tin đồn", "cực kỳ nguy hiểm"]

    texts, labels = [], []
    for _ in range(n // 2):
        t = " ".join(rng.choice(real_words, size=5, replace=True))
        texts.append(t)
        labels.append(0)
    for _ in range(n // 2):
        t = " ".join(rng.choice(fake_words, size=5, replace=True))
        texts.append(t)
        labels.append(1)

    # Shuffle so that train/test split gets both classes
    idx = rng.permutation(n)
    df = pd.DataFrame({"text": [texts[i] for i in idx], "label": [labels[i] for i in idx]})
    return df


class TestEndToEndPipeline:
    """Feeds synthetic data through TF-IDF → LR → evaluate."""

    def test_tfidf_lr_pipeline(self, synthetic_dataset):
        df = synthetic_dataset

        # Split (stratified to ensure both classes in each set)
        train_df = df.iloc[:70]
        test_df = df.iloc[70:]

        # Feature extraction
        ext = TfidfFeatureExtractor(max_features=200, min_df=1)
        X_train = ext.fit_transform(train_df["text"])
        X_test = ext.transform(test_df["text"])

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # Train
        trainer = LogisticRegressionTrainer(class_weight="balanced", random_state=42)
        trainer.train(X_train, y_train)

        # Predict
        y_pred = trainer.predict(X_test)
        y_prob = trainer.predict_proba(X_test)

        # Evaluate
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Sanity: the trivially separable dataset should yield decent accuracy
        assert metrics["accuracy"] >= 0.7
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics

    def test_save_load_predict_consistency(self, synthetic_dataset, tmp_path):
        """Train → save → load → predict must produce identical results."""
        df = synthetic_dataset
        ext = TfidfFeatureExtractor(max_features=200, min_df=1)
        X = ext.fit_transform(df["text"])
        y = df["label"].values

        trainer = LogisticRegressionTrainer(class_weight="balanced", random_state=42)
        trainer.train(X, y)
        preds_before = trainer.predict(X)

        path = str(tmp_path / "lr.pkl")
        trainer.save(path)
        loaded = LogisticRegressionTrainer.load(path)
        preds_after = loaded.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)
