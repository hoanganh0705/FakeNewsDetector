"""
Shared logging utility for FakeNewsDetector.

Provides a single get_logger() factory so every module gets a consistently
formatted logger instead of using bare print() calls.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Loading data from %s", path)
    log.warning("No predictions found for %s", model_name)
    log.error("Training failed: %s", e)
"""

import logging
import sys
from typing import Optional


# Global format — timestamp | level | module name | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger name for the whole project
_ROOT = "fakenews"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger attached to the project root logger.

    The root logger writes INFO+ to stdout. Call this once per module:

        log = get_logger(__name__)

    Args:
        name:  Module name (pass __name__ from the calling module).
        level: Logging level for this specific logger (default: INFO).

    Returns:
        A configured logging.Logger instance.
    """
    # Configure the root project logger exactly once
    root = logging.getLogger(_ROOT)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)   # root captures everything; handlers filter

    logger = logging.getLogger(f"{_ROOT}.{name}")
    logger.setLevel(level)
    return logger


def set_global_level(level: int) -> None:
    """
    Change the log level for every logger in the project at runtime.

    Useful to silence all output during batch jobs:
        set_global_level(logging.WARNING)

    Args:
        level: A logging level constant (e.g. logging.DEBUG, logging.WARNING).
    """
    logging.getLogger(_ROOT).setLevel(level)
