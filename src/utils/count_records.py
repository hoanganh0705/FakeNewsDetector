import os
import pandas as pd


def count_records(path: str, label_col: str = "label", text_col: str = "text"):
    """
    Count number of records in any CSV file.
    - Prints total rows
    - Prints missing values
    - If label_col exists: prints label counts + ratio
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    print("\n==============================")
    print(f"📄 FILE: {path}")
    print("==============================")

    print(f"✅ Total rows: {len(df)}")
    print(f"✅ Total columns: {len(df.columns)}")
    print("🧾 Columns:", list(df.columns))

    # Missing values report
    print("\n==== MISSING VALUES ====")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"⚠️ {col}: {missing}")

    if df.isna().sum().sum() == 0:
        print("✅ No missing values")

    # Text stats (if exists)
    if text_col in df.columns:
        empty_text = df[text_col].astype(str).str.strip().eq("").sum()
        print("\n==== TEXT INFO ====")
        print(f"Empty '{text_col}' rows: {empty_text}")

    # Label stats (if exists)
    if label_col in df.columns:
        print("\n==== LABEL COUNTS ====")
        counts = df[label_col].value_counts(dropna=False)
        print(counts)

        # ratio calc (only if binary)
        unique_labels = counts.index.tolist()
        if len(unique_labels) == 2:
            a, b = unique_labels[0], unique_labels[1]
            ca, cb = counts[a], counts[b]
            if cb != 0:
                print(f"\n📊 Ratio {a}:{b} = 1 : {cb/ca:.2f}")
        else:
            print("\nℹ️ Label column exists but not binary (not 2 classes).")

    print("==============================\n")


if __name__ == "__main__":
    # Example usage
    count_records("data/raw/raw.csv")
    count_records("data/splits/train.csv")
    count_records("data/splits/val.csv")
    count_records("data/splits/test.csv")
