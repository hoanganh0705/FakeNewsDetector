import os
import re
import pandas as pd
from tqdm import tqdm

from config import cfg, ROOT_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

__all__ = [
    "normalize_text",
    "segment_text",
    "process_column",
]

_segmenter = None
_segmenter_name = ""
_segmenter_initialized = False
_ut_tokenize = None  # will be set by _init_segmenter() if underthesea is used


def _init_segmenter():
    global _segmenter, _segmenter_name, _segmenter_initialized, _ut_tokenize
    if _segmenter_initialized:
        return
    _segmenter_initialized = True

    try:
        import py_vncorenlp  # type: ignore
        _SAVE_DIR = os.path.join(ROOT_DIR, ".vncorenlp")
        os.makedirs(_SAVE_DIR, exist_ok=True)
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

RAW_DATA_PATH = cfg.PATHS.raw_data
PROCESSED_DIR = os.path.dirname(cfg.PATHS.segmented_data)
OUTPUT_PATH   = cfg.PATHS.segmented_data


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segment_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    _init_segmenter()
    text = normalize_text(text)

    try:
        if _segmenter_name.startswith("py_vncorenlp"):
            sentences = _segmenter.word_segment(text)
            return " ".join(sentences)
        else:
            return _ut_tokenize(text, format="text")
    except (RuntimeError, ValueError, TypeError) as e:
        log.warning("Segmentation error (falling back to raw text): %s", e)
        return text



def process_column(series: pd.Series, desc: str) -> pd.Series:
    results = []
    for text in tqdm(series, desc=desc):
        results.append(segment_text(text))
    return pd.Series(results, index=series.index)


def main():
    log.info("=" * 60)
    log.info("Vietnamese Word Segmentation")
    log.info("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    log.info("Loading data from %s...", RAW_DATA_PATH)
    from src.utils.common import load_csv
    df = load_csv(RAW_DATA_PATH, required_columns=['text'])
    log.info("Total records: %d", len(df))

    log.info("Columns: %s", list(df.columns))

    sample_idx = 0
    sample_original_text = df.iloc[sample_idx]['text'] if len(df) > 0 else ""

    log.info("Applying word segmentation...")

    log.info("Processing 'text' column...")
    df['text'] = process_column(df['text'], "   Segmenting texts")

    log.info("Saving segmented data to %s...", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)

    log.info("Sample of segmented text:")
    log.info("-" * 60)
    log.info("Original text: %s...", str(sample_original_text)[:100])
    log.info("Segmented text: %s...", str(df.iloc[sample_idx]['text'])[:100] if len(df) > 0 else "")
    log.info("-" * 60)

    log.info("Segmentation Statistics:")
    log.info("Total records processed: %d", len(df))
    log.info("Output file: %s", OUTPUT_PATH)

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
