"""
Vietnamese Word Segmentation Script.
Applies consistent word segmentation to the raw dataset using underthesea.
"""

import os
import sys
import re
import pandas as pd
from tqdm import tqdm
from underthesea import word_tokenize

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'raw.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'segmented.csv')


def normalize_text(text: str) -> str:
    """
    Normalize text before segmentation.
    - Remove existing underscores from previous segmentation
    - Clean up whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Replace underscores with spaces (remove previous segmentation)
    text = text.replace('_', ' ')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def segment_text(text: str) -> str:
    """
    Apply Vietnamese word segmentation using underthesea.
    
    Args:
        text: Input text
        
    Returns:
        Segmented text with underscores connecting word compounds
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Normalize first
    text = normalize_text(text)
    
    try:
        # Apply word segmentation
        # format="text" returns segmented string with underscores
        segmented = word_tokenize(text, format="text")
        return segmented
    except Exception as e:
        print(f"Error segmenting text: {e}")
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
