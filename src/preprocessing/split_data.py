"""
Script to split raw data into train, validation, and test sets.
- Train: 70%
- Validation: 15%
- Test: 15%

Uses stratified splitting to ensure even distribution of labels.
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

def main():
    # Create splits directory if it doesn't exist
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    # Load the raw data
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # First split: separate test set (15%)
    # Remaining 85% will be split into train and validation
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=df['label'],
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
    
    # Save the splits
    train_path = os.path.join(SPLITS_DIR, 'train.csv')
    val_path = os.path.join(SPLITS_DIR, 'val.csv')
    test_path = os.path.join(SPLITS_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("Data Split Summary")
    print("="*50)
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Label 0 (Real): {len(train_df[train_df['label']==0])}")
    print(f"  - Label 1 (Fake): {len(train_df[train_df['label']==1])}")
    print(f"\nValidation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Label 0 (Real): {len(val_df[val_df['label']==0])}")
    print(f"  - Label 1 (Fake): {len(val_df[val_df['label']==1])}")
    print(f"\nTest set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  - Label 0 (Real): {len(test_df[test_df['label']==0])}")
    print(f"  - Label 1 (Fake): {len(test_df[test_df['label']==1])}")
    print("="*50)
    print(f"\nFiles saved to: {SPLITS_DIR}")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")

if __name__ == "__main__":
    main()
