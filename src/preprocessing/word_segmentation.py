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
import sys
import re
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Pick the best available segmenter ──────────────────────────────────────────
# Priority 1: py_vncorenlp  (VinAI RDRSegmenter, same as PhoBERT training)
# Priority 2: underthesea   (pure-Python fallback, easier install)

_segmenter = None
_segmenter_name = ""

try:
    import py_vncorenlp  # type: ignore
    _SAVE_DIR = os.path.join(PROJECT_ROOT, ".vncorenlp")
    os.makedirs(_SAVE_DIR, exist_ok=True)
    # Download model weights on first run (cached afterwards)
    py_vncorenlp.download_model(save_dir=_SAVE_DIR)
    _segmenter = py_vncorenlp.VnCoreNLP(save_dir=_SAVE_DIR, annotators=["wseg"])
    _segmenter_name = "py_vncorenlp (RDRSegmenter)"
    log.info("Using segmenter: %s", _segmenter_name)
except Exception as e:
    log.warning("py_vncorenlp unavailable (%s). Falling back to underthesea.", e)
    try:
        from underthesea import word_tokenize as _ut_tokenize  # type: ignore
        _segmenter_name = "underthesea"
        log.info("Using segmenter: %s", _segmenter_name)
    except ImportError:
        log.error("Neither py_vncorenlp nor underthesea is installed. "
                  "Run: pip install py_vncorenlp")
        raise

# Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'raw.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_PATH   = os.path.join(PROCESSED_DIR, 'segmented.csv')


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

    text = normalize_text(text)

    try:
        if _segmenter_name.startswith("py_vncorenlp"):
            # word_segment() returns list[str]  (each str is a segmented sentence)
            sentences = _segmenter.word_segment(text)
            return " ".join(sentences)
        else:
            # underthesea: format="text" returns underscore-joined string
            return _ut_tokenize(text, format="text")  # type: ignore[name-defined]
    except Exception as e:
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
    print("\n" + "=" * 60)
    print(" Vietnamese Word Segmentation")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Load raw data
    print(f"\n📂 Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Total records: {len(df)}")
    
    # Check columns
    print(f"   Columns: {list(df.columns)}")
    
    # Apply segmentation to title and text columns
    print("\n🔧 Applying word segmentation...")
    
    # Process title column
    print("   Processing 'title' column...")
    df['title'] = process_column(df['title'], "   Segmenting titles")
    
    # Process text column
    print("   Processing 'text' column...")
    df['text'] = process_column(df['text'], "   Segmenting texts")
    
    # Save processed data
    print(f"\n💾 Saving segmented data to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Show sample
    print("\n📋 Sample of segmented text:")
    print("-" * 60)
    original_df = pd.read_csv(RAW_DATA_PATH)
    sample_idx = 0
    print(f"Original title: {original_df.iloc[sample_idx]['title'][:100]}...")
    print(f"Segmented title: {df.iloc[sample_idx]['title'][:100]}...")
    print("-" * 60)
    
    # Statistics
    print("\n📊 Segmentation Statistics:")
    print(f"   Total records processed: {len(df)}")
    print(f"   Output file: {OUTPUT_PATH}")
    
    # Check for common Vietnamese compound words
    sample_text = df['text'].iloc[0]
    common_compounds = ['việt_nam', 'trung_quốc', 'thành_phố', 'chính_phủ', 'xã_hội']
    found = [w for w in common_compounds if w in sample_text.lower()]
    if found:
        print(f"   Sample compounds found: {found}")
    
    print("\n✅ Word segmentation complete!")
    print("=" * 60 + "\n")
    
    return df


if __name__ == '__main__':
    main()
