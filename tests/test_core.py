"""
Backward-compatibility shim.

Tests formerly defined here have been split into dedicated modules:
  - test_metrics.py
  - test_text_preprocessor.py
  - test_statistical_tests.py
  - test_config.py

This file re-exports all classes so that ``pytest tests/test_core.py``
continues to work unchanged.
"""

from tests.test_metrics import TestComputeMetrics                          # noqa: F401
from tests.test_text_preprocessor import TestTextPreprocessor              # noqa: F401
from tests.test_statistical_tests import TestMcNemarTest, TestHolmBonferroni  # noqa: F401
from tests.test_config import TestConfig                                   # noqa: F401
