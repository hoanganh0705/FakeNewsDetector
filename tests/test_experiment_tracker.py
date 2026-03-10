"""Tests for ExperimentTracker versioning utility."""

import json
import os

import pytest

from src.utils.common import ExperimentTracker


class TestExperimentTracker:

    def test_log_run_creates_file(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        tracker.log_run(
            model_name="TestModel",
            config={"lr": 0.01},
            metrics={"accuracy": 0.95, "f1_macro": 0.93},
        )
        assert os.path.exists(tracker.log_path)

    def test_log_run_appends(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        tracker.log_run("M1", {"a": 1}, {"accuracy": 0.9, "f1_macro": 0.8})
        tracker.log_run("M1", {"a": 2}, {"accuracy": 0.91, "f1_macro": 0.82})

        with open(tracker.log_path) as f:
            records = json.load(f)
        assert len(records) == 2

    def test_record_fields(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        record = tracker.log_run(
            "LR",
            {"C": 1.0},
            {"accuracy": 0.88, "f1_macro": 0.85, "precision_macro": 0.86, "recall_macro": 0.84},
        )
        assert "run_id" in record
        assert "timestamp" in record
        assert "config_hash" in record
        assert record["model"] == "LR"
        assert record["metrics_summary"]["accuracy"] == 0.88

    def test_config_hash_deterministic(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        r1 = tracker.log_run("M", {"x": 1}, {"accuracy": 0.5, "f1_macro": 0.5})
        r2 = tracker.log_run("M", {"x": 1}, {"accuracy": 0.6, "f1_macro": 0.6})
        assert r1["config_hash"] == r2["config_hash"]

    def test_different_config_different_hash(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        r1 = tracker.log_run("M", {"x": 1}, {"accuracy": 0.5, "f1_macro": 0.5})
        r2 = tracker.log_run("M", {"x": 2}, {"accuracy": 0.5, "f1_macro": 0.5})
        assert r1["config_hash"] != r2["config_hash"]

    def test_extra_field_stored(self, tmp_path):
        tracker = ExperimentTracker(str(tmp_path))
        record = tracker.log_run(
            "M", {"a": 1}, {"accuracy": 0.5, "f1_macro": 0.5},
            extra={"note": "test run"},
        )
        assert record["extra"]["note"] == "test run"
