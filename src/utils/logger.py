
import logging
import sys
import warnings
from typing import Optional


warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn(\.|$)")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"scipy(\.|$)")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"joblib(\.|$)")

__all__ = ["get_logger", "set_global_level"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_ROOT = "fakenews"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger(_ROOT)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)

    logger = logging.getLogger(f"{_ROOT}.{name}")
    logger.setLevel(level)
    return logger


def set_global_level(level: int) -> None:
    logging.getLogger(_ROOT).setLevel(level)
