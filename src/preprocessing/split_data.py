"""
Script to split raw data into train, validation, and test sets.
- Train: 70%
- Validation: 15%
- Test: 15%

Uses stratified splitting to ensure even distribution of labels.
Includes data cleaning to remove duplicates and prevent data leakage.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuration
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'segmented.csv')
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicates and invalid entries.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    original_len = len(df)
    
    print("\n" + "="*50)
    print("🧹 DATA CLEANING")
    print("="*50)
    
    # 1. Remove exact duplicate texts (keep first occurrence)
    df_clean = df.drop_duplicates(subset=['text'], keep='first')
    duplicates_removed = original_len - len(df_clean)
    print(f"✅ Removed {duplicates_removed} duplicate texts")
    
    # 2. Remove rows with empty or NaN text
    df_clean = df_clean.dropna(subset=['text'])
    df_clean = df_clean[df_clean['text'].astype(str).str.strip() != '']
    empty_removed = original_len - duplicates_removed - len(df_clean)
    if empty_removed > 0:
        print(f"✅ Removed {empty_removed} empty texts")
    
    # 3. Remove very short texts (less than 5 words) - likely noise
    df_clean['word_count'] = df_clean['text'].astype(str).apply(lambda x: len(x.split()))
    short_texts = (df_clean['word_count'] < 5).sum()
    df_clean = df_clean[df_clean['word_count'] >= 5]
    df_clean = df_clean.drop(columns=['word_count'])
    if short_texts > 0:
        print(f"✅ Removed {short_texts} very short texts (<5 words)")
    
    # 4. Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"\n📊 Dataset size: {original_len} → {len(df_clean)} samples")
    print(f"   Total removed: {original_len - len(df_clean)} samples")
    
    return df_clean


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Verify there is no data leakage between splits.
    
    Returns:
        True if no leakage detected, False otherwise
    """
    train_texts = set(train_df['text'].astype(str))
    val_texts = set(val_df['text'].astype(str))
    test_texts = set(test_df['text'].astype(str))
    
    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))
    
    print("\n" + "="*50)
    print("🔍 DATA LEAKAGE CHECK")
    print("="*50)
    print(f"  Train-Val overlap: {train_val_overlap}")
    print(f"  Train-Test overlap: {train_test_overlap}")
    print(f"  Val-Test overlap: {val_test_overlap}")
    
    if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
        print("  ✅ No data leakage detected!")
        return True
    else:
        print("  ❌ Data leakage detected!")
        return False


def main():
    # Create splits directory if it doesn't exist
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    # Load the raw data
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print(f"Total samples loaded: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Clean the data first
    df_clean = clean_data(df)
    
    print("\n" + "="*50)
    print("📂 SPLITTING DATA")
    print("="*50)
    print(f"Clean dataset size: {len(df_clean)}")
    print(f"Label distribution after cleaning:\n{df_clean['label'].value_counts()}")
    
    # First split: separate test set (15%)
    # Remaining 85% will be split into train and validation
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=df_clean['label'],
        shuffle=True
    )
    
    # Second split: separate validation set from train_val
    # val_ratio / (train_ratio + val_ratio) = 0.15 / 0.85 ≈ 0.176
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE,
        stratify=train_val_df['label'],
        shuffle=True
    )
    
    # Verify no data leakage
    verify_no_leakage(train_df, val_df, test_df)
    
    # Save the splits
    train_path = os.path.join(SPLITS_DIR, 'train.csv')
    val_path = os.path.join(SPLITS_DIR, 'val.csv')
    test_path = os.path.join(SPLITS_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DATA SPLIT SUMMARY")
    print("="*50)
    
    total_clean = len(df_clean)
    
    print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/total_clean*100:.1f}%)")
    print(f"  - Label 0 (Real): {len(train_df[train_df['label']==0])}")
    print(f"  - Label 1 (Fake): {len(train_df[train_df['label']==1])}")
    balance_train = min(len(train_df[train_df['label']==0]), len(train_df[train_df['label']==1])) / max(len(train_df[train_df['label']==0]), len(train_df[train_df['label']==1]))
    print(f"  - Balance ratio: {balance_train:.2f}")
    
    print(f"\nValidation set: {len(val_df)} samples ({len(val_df)/total_clean*100:.1f}%)")
    print(f"  - Label 0 (Real): {len(val_df[val_df['label']==0])}")
    print(f"  - Label 1 (Fake): {len(val_df[val_df['label']==1])}")
    
    print(f"\nTest set: {len(test_df)} samples ({len(test_df)/total_clean*100:.1f}%)")
    print(f"  - Label 0 (Real): {len(test_df[test_df['label']==0])}")
    print(f"  - Label 1 (Fake): {len(test_df[test_df['label']==1])}")
    
    print("\n" + "="*50)
    print(f"✅ Files saved to: {SPLITS_DIR}")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print("="*50)


if __name__ == "__main__":
    main()
