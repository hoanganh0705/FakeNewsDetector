"""
Vietnamese Word Segmentation Script.

Uses py_vncorenlp (VinAI's RDRSegmenter) as the primary segmenter — this is
the SAME tokenizer PhoBERT was trained with, so it produces optimal compound-
word labels (e.g. "Việt_Nam", "thành_phố") that align with the model's
vocabulary.

Falls back to underthesea automatically if py_vncorenlp / Java is not
available, so the script never hard-fails.

Install (once):
    pip install py_vncorenlp

py_vncorenlp bundles the VnCoreNLP JAR and downloads the model weights on
first run — no separate Java setup required.
"""

import os
import re
import pandas as pd
from tqdm import tqdm

from config import cfg, ROOT_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Pick the best available segmenter ──────────────────────────────────────────
# Priority 1: py_vncorenlp  (VinAI RDRSegmenter, same as PhoBERT training)
# Priority 2: underthesea   (pure-Python fallback, easier install)

_segmenter = None
_segmenter_name = ""
_segmenter_initialized = False
_ut_tokenize = None  # will be set by _init_segmenter() if underthesea is used


def _init_segmenter():
    """Lazy-initialize the word segmenter on first use (avoids heavy import at module load)."""
    global _segmenter, _segmenter_name, _segmenter_initialized, _ut_tokenize
    if _segmenter_initialized:
        return
    _segmenter_initialized = True

    try:
        import py_vncorenlp  # type: ignore
        _SAVE_DIR = os.path.join(ROOT_DIR, ".vncorenlp")
        os.makedirs(_SAVE_DIR, exist_ok=True)
        # Download model weights on first run (cached afterwards)
        py_vncorenlp.download_model(save_dir=_SAVE_DIR)
        _segmenter = py_vncorenlp.VnCoreNLP(save_dir=_SAVE_DIR, annotators=["wseg"])
        _segmenter_name = "py_vncorenlp (RDRSegmenter)"
        log.info("Using segmenter: %s", _segmenter_name)
    except (ImportError, OSError, RuntimeError) as e:
        log.warning("py_vncorenlp unavailable (%s). Falling back to underthesea.", e)
        try:
            from underthesea import word_tokenize as _ut_tokenize  # type: ignore  # noqa: F811
            _segmenter_name = "underthesea"
            log.info("Using segmenter: %s", _segmenter_name)
        except ImportError:
            log.error("Neither py_vncorenlp nor underthesea is installed. "
                      "Run: pip install py_vncorenlp")
            raise

# Paths (derived from central config — single source of truth)
RAW_DATA_PATH = cfg.PATHS.raw_data
PROCESSED_DIR = os.path.dirname(cfg.PATHS.segmented_data)
OUTPUT_PATH   = cfg.PATHS.segmented_data


def normalize_text(text: str) -> str:
    """
    Normalize text before segmentation.
    - Remove existing underscores from previous segmentation
    - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segment_text(text: str) -> str:
    """
    Apply Vietnamese word segmentation to a single string.

    Uses py_vncorenlp (RDRSegmenter) when available — the same tokenizer
    PhoBERT was trained on — falling back to underthesea otherwise.

    Args:
        text: Raw Vietnamese text

    Returns:
        Segmented text with underscores connecting compound words,
        e.g. "Chính phủ" → "Chính_phủ"
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    _init_segmenter()
    text = normalize_text(text)

    try:
        if _segmenter_name.startswith("py_vncorenlp"):
            # word_segment() returns list[str]  (each str is a segmented sentence)
            sentences = _segmenter.word_segment(text)
            return " ".join(sentences)
        else:
            # underthesea: format="text" returns underscore-joined string
            return _ut_tokenize(text, format="text")
    except (RuntimeError, ValueError, TypeError) as e:
        log.warning("Segmentation error (falling back to raw text): %s", e)
        return text



def process_column(series: pd.Series, desc: str) -> pd.Series:
    """Process a pandas series with progress bar."""
    results = []
    for text in tqdm(series, desc=desc):
        results.append(segment_text(text))
    return pd.Series(results, index=series.index)


def main():
    """Main function to process the dataset."""
    log.info("=" * 60)
    log.info("Vietnamese Word Segmentation")
    log.info("=" * 60)

    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw data
    log.info("Loading data from %s...", RAW_DATA_PATH)
    from src.utils.common import load_csv
    df = load_csv(RAW_DATA_PATH, required_columns=['text'])
    log.info("Total records: %d", len(df))

    # Check columns
    log.info("Columns: %s", list(df.columns))

    sample_idx = 0
    sample_original_text = df.iloc[sample_idx]['text'] if len(df) > 0 else ""

    # Apply segmentation to text column only (dataset: id, text, date, label)
    log.info("Applying word segmentation...")

    # Process text column
    log.info("Processing 'text' column...")
    df['text'] = process_column(df['text'], "   Segmenting texts")

    # Save processed data
    log.info("Saving segmented data to %s...", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)

    # Show sample
    log.info("Sample of segmented text:")
    log.info("-" * 60)
    log.info("Original text: %s...", str(sample_original_text)[:100])
    log.info("Segmented text: %s...", str(df.iloc[sample_idx]['text'])[:100] if len(df) > 0 else "")
    log.info("-" * 60)

    # Statistics
    log.info("Segmentation Statistics:")
    log.info("Total records processed: %d", len(df))
    log.info("Output file: %s", OUTPUT_PATH)

    # Check for common Vietnamese compound words
    sample_text = df['text'].iloc[0]
    common_compounds = ['việt_nam', 'trung_quốc', 'thành_phố', 'chính_phủ', 'xã_hội']
    found = [w for w in common_compounds if w in sample_text.lower()]
    if found:
        log.info("Sample compounds found: %s", found)

    log.info("Word segmentation complete!")
    log.info("=" * 60)

    return df


if __name__ == '__main__':
    main()
