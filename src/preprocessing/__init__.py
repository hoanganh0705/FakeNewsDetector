# Preprocessing module
from .text_preprocessor import TextPreprocessor
from .split_data import main as split_data

# Lazy import for load_data to avoid pulling in heavy dependencies at import time
def load_data(*args, **kwargs):
    from .text_preprocessor import load_data as _load_data
    return _load_data(*args, **kwargs)

__all__ = ['TextPreprocessor', 'load_data', 'split_data']
