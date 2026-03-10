"""Tests for data splitting (src.preprocessing.split_data)."""

import pandas as pd
import pytest

from src.preprocessing.split_data import verify_no_leakage


class TestVerifyNoLeakage:
    """Ensure the leakage detection utility works."""

    def test_no_leakage_passes(self):
        train = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
        val = pd.DataFrame({"text": ["c", "d"], "label": [0, 1]})
        test = pd.DataFrame({"text": ["e", "f"], "label": [0, 1]})
        result = verify_no_leakage(train, val, test)
        assert result is True

    def test_leakage_detected(self):
        """Overlapping text between train and test should be caught."""
        train = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        val = pd.DataFrame({"text": ["d"], "label": [1]})
        test = pd.DataFrame({"text": ["a", "e"], "label": [0, 1]})  # 'a' overlaps
        result = verify_no_leakage(train, val, test)
        assert result is False, "Expected leakage to be detected (overlapping 'a')"
