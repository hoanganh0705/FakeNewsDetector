import os
import pandas as pd

FILES_TO_CLEAN = [
    "data/raw/raw.csv",
    "data/splits/train.csv",
]

def clean_file(path: str):
    if not os.path.exists(path):
        print(f"⚠️ Skip (not found): {path}")
        return

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "text" not in df.columns:
        print(f"⚠️ No 'text' column found in {path}")
        return

    before = len(df)

    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # drop empty text
    df = df[df["text"].str.len() > 0].copy()

    after = len(df)
    removed = before - after

    df.to_csv(path, index=False)

    print(f"✅ Cleaned: {path}")
    print(f"Before: {before} | After: {after} | Removed empty text: {removed}\n")


def main():
    for f in FILES_TO_CLEAN:
        clean_file(f)

if __name__ == "__main__":
    main()
