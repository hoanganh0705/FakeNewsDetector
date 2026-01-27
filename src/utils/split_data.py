import os
import pandas as pd

RAW_DATA_PATH = "data/raw/dataset_clean_final.csv"
SPLIT_DIR = "data/splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def split_time_based(
    raw_path=RAW_DATA_PATH,
    split_dir=SPLIT_DIR,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

    os.makedirs(split_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["id", "title", "text", "date", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Clean
    df = df.dropna(subset=["id", "text", "label", "date"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)

    # Parse date: format YYYY-MM-DD (your dataset matches this)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date_parsed"]).copy()

    # Sort by time
    df = df.sort_values("date_parsed").reset_index(drop=True)

    total = len(df)
    if total < 50:
        raise ValueError(f"Dataset too small after cleaning: {total} rows")

    # Compute split sizes
    test_size = int(round(total * test_ratio))
    val_size = int(round(total * val_ratio))
    train_size = total - val_size - test_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Invalid split sizes. Check ratios.")

    # Split by time (pattern-based)
    train = df.iloc[:train_size].copy()
    val = df.iloc[train_size:train_size + val_size].copy()
    test = df.iloc[train_size + val_size:].copy()

    # Leakage check by ID
    train_ids = set(train["id"])
    val_ids = set(val["id"])
    test_ids = set(test["id"])

    assert train_ids.isdisjoint(val_ids), "Leakage: train overlaps val (same IDs)"
    assert train_ids.isdisjoint(test_ids), "Leakage: train overlaps test (same IDs)"
    assert val_ids.isdisjoint(test_ids), "Leakage: val overlaps test (same IDs)"

    # Report date ranges
    print("==== TIME-BASED SPLIT (LATEST 15% TEST) ====")
    print("Total:", total)
    print("Train:", len(train), "| date:", train["date_parsed"].min(), "->", train["date_parsed"].max())
    print("Val  :", len(val), "| date:", val["date_parsed"].min(), "->", val["date_parsed"].max())
    print("Test :", len(test), "| date:", test["date_parsed"].min(), "->", test["date_parsed"].max())

    print("\n==== LABEL DISTRIBUTION (counts) ====")
    print("Train:\n", train["label"].value_counts())
    print("Val:\n", val["label"].value_counts())
    print("Test:\n", test["label"].value_counts())

    print("\n==== LABEL DISTRIBUTION (ratio) ====")
    print("Train:\n", train["label"].value_counts(normalize=True))
    print("Val:\n", val["label"].value_counts(normalize=True))
    print("Test:\n", test["label"].value_counts(normalize=True))

    # Drop helper column before saving
    train = train.drop(columns=["date_parsed"])
    val = val.drop(columns=["date_parsed"])
    test = test.drop(columns=["date_parsed"])

    # Save
    train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    print("\n✅ Saved:")
    print("-", os.path.join(split_dir, "train.csv"))
    print("-", os.path.join(split_dir, "val.csv"))
    print("-", os.path.join(split_dir, "test.csv"))


if __name__ == "__main__":
    split_time_based()
