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

from config import cfg
from src.preprocessing.text_preprocessor import clean_dataset
from src.utils.common import load_csv

from src.utils.logger import get_logger
log = get_logger(__name__)


# Pull settings from central config
RANDOM_STATE = cfg.RANDOM_STATE
TRAIN_RATIO  = cfg.DATA.train_ratio
VAL_RATIO    = cfg.DATA.val_ratio
TEST_RATIO   = cfg.DATA.test_ratio

# Paths
RAW_DATA_PATH = cfg.PATHS.segmented_data
SPLITS_DIR   = cfg.PATHS.splits_dir


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicates and invalid entries.

    Delegates to the canonical ``clean_dataset()`` in ``text_preprocessor``
    so that there is a single source-of-truth for data cleaning logic.
    """
    return clean_dataset(df, min_words=cfg.DATA.min_word_count)


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Verify there is no data leakage between splits.
    Uses both text-content comparison and id-based comparison.

    Returns:
        True if no leakage detected, False otherwise
    """


    log.info("\n" + "="*50)
    log.info("DATA LEAKAGE CHECK")
    log.info("="*50)

    # ── Text-content check ────────────────────────────────────────
    train_texts = set(train_df['text'].astype(str))
    val_texts   = set(val_df['text'].astype(str))
    test_texts  = set(test_df['text'].astype(str))

    train_val_overlap   = len(train_texts & val_texts)
    train_test_overlap  = len(train_texts & test_texts)
    val_test_overlap    = len(val_texts   & test_texts)

    log.info("  [Text-based]")
    log.info(f"    Train-Val overlap:  {train_val_overlap}")
    log.info(f"    Train-Test overlap: {train_test_overlap}")
    log.info(f"    Val-Test overlap:   {val_test_overlap}")

    text_clean = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)

    # ── ID-based check (exact, unaffected by segmentation) ────────
    id_clean = True
    if 'id' in train_df.columns:
        train_ids = set(train_df['id'].astype(str))
        val_ids   = set(val_df['id'].astype(str))
        test_ids  = set(test_df['id'].astype(str))

        id_tv  = len(train_ids & val_ids)
        id_tt  = len(train_ids & test_ids)
        id_vt  = len(val_ids   & test_ids)

        log.info("  [ID-based]")
        log.info(f"    Train-Val ID overlap:  {id_tv}")
        log.info(f"    Train-Test ID overlap: {id_tt}")
        log.info(f"    Val-Test ID overlap:   {id_vt}")

        id_clean = (id_tv == 0 and id_tt == 0 and id_vt == 0)
    else:
        log.warning("  [ID-based]   'id' column not found, skipping")

    if text_clean and id_clean:
        log.info("No data leakage detected!")
        return True
    else:
        log.error("Data leakage detected!")
        return False


def main():
    # Create splits directory if it doesn't exist
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Load the raw data
    log.info("Loading data from %s...", RAW_DATA_PATH)
    df = load_csv(RAW_DATA_PATH, required_columns=['text', 'label'])

    log.info("Total samples loaded: %d", len(df))
    log.info("Label distribution:\n%s", df['label'].value_counts())

    # Clean the data first
    df_clean = clean_data(df)

    log.info("=" * 50)
    log.info("SPLITTING DATA")
    log.info("=" * 50)
    log.info("Clean dataset size: %d", len(df_clean))
    log.info("Label distribution after cleaning:\n%s", df_clean['label'].value_counts())
    
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
    log.info("=" * 50)
    log.info("DATA SPLIT SUMMARY")
    log.info("=" * 50)

    total_clean = len(df_clean)

    log.info("Train set: %d samples (%.1f%%)", len(train_df), len(train_df)/total_clean*100)
    log.info("  - Label 0 (Real): %d", len(train_df[train_df['label']==0]))
    log.info("  - Label 1 (Fake): %d", len(train_df[train_df['label']==1]))
    balance_train = min(len(train_df[train_df['label']==0]), len(train_df[train_df['label']==1])) / max(len(train_df[train_df['label']==0]), len(train_df[train_df['label']==1]))
    log.info("  - Balance ratio: %.2f", balance_train)

    log.info("Validation set: %d samples (%.1f%%)", len(val_df), len(val_df)/total_clean*100)
    log.info("  - Label 0 (Real): %d", len(val_df[val_df['label']==0]))
    log.info("  - Label 1 (Fake): %d", len(val_df[val_df['label']==1]))

    log.info("Test set: %d samples (%.1f%%)", len(test_df), len(test_df)/total_clean*100)
    log.info("  - Label 0 (Real): %d", len(test_df[test_df['label']==0]))
    log.info("  - Label 1 (Fake): %d", len(test_df[test_df['label']==1]))

    log.info("=" * 50)
    log.info("Files saved to: %s", SPLITS_DIR)
    log.info("  - %s", train_path)
    log.info("  - %s", val_path)
    log.info("  - %s", test_path)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
