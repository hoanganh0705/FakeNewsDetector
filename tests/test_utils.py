"""Tests for CSV loading and validation utilities."""

import os

import pandas as pd
import pytest

from src.utils.common import load_csv, validate_dataframe_columns


class TestValidateDataframeColumns:

    def test_passes_when_columns_present(self):
        df = pd.DataFrame({"text": ["a"], "label": [0]})
        validate_dataframe_columns(df, ["text", "label"])

    def test_raises_when_column_missing(self):
        df = pd.DataFrame({"text": ["a"]})
        with pytest.raises(ValueError, match="missing"):
            validate_dataframe_columns(df, ["text", "label"])


class TestLoadCsv:

    def test_loads_valid_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        pd.DataFrame({"text": ["hello"], "label": [1]}).to_csv(path, index=False)
        df = load_csv(path)
        assert len(df) == 1

    def test_validates_columns(self, tmp_path):
        path = str(tmp_path / "data.csv")
        pd.DataFrame({"text": ["hello"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing"):
            load_csv(path, required_columns=["text", "label"])

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")
