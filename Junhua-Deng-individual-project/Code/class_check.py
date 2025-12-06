import pandas as pd
import numpy as np
import os

CSV_PATH = "./data/review_clean.csv"   # 修改路径请改这里

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, encoding="latin1")

    if "Rating" not in df.columns:
        print("❌ No 'Rating' column found in CSV.")
        return

    # 去除缺失
    df = df.dropna(subset=["Rating"])
    df["Rating"] = df["Rating"].astype(int)

    print("=== Rating Class Distribution ===")
    counts = df["Rating"].value_counts().sort_index()
    print(counts)

    print("\n=== Class Ratio ===")
    ratios = df["Rating"].value_counts(normalize=True).sort_index()
    print(ratios)

    # imbalance score
    max_count = counts.max()
    min_count = counts.min()
    ratio = max_count / min_count if min_count > 0 else float("inf")

    print("\n=== Imbalance Score (max / min) ===")
    print(f"{ratio:.2f}")

    if ratio <= 3:
        print("➡️ Mild or no imbalance")
    elif ratio <= 10:
        print("➡️ Moderate imbalance")
    else:
        print("➡️ Severe imbalance")


if __name__ == "__main__":
    main()
