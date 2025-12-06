import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. 读取 CSV
    df = pd.read_csv("review_raw.csv", encoding="latin1")

    df["Year_Month"] = df["Year_Month"].astype(str).str.strip()
    invalid_values = ["missing", "nan", "none", "null", "", " "]
    df["Year_Month"] = df["Year_Month"].replace(invalid_values, pd.NA)
    df = df.dropna(subset=["Year_Month"])
    df["Year_Month"] = pd.to_datetime(df["Year_Month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["Year_Month"])

    # 2. 计算每个 Branch 每个月的平均 Rating
    monthly_avg = (
        df.groupby(["Year_Month", "Branch"])["Rating"]
        .mean()
        .reset_index()
        .sort_values("Year_Month")
    )

    # 3. 绘制折线图
    plt.figure(figsize=(12, 7))

    branches = monthly_avg["Branch"].unique()

    for b in branches:
        subset = monthly_avg[monthly_avg["Branch"] == b]
        plt.plot(
            subset["Year_Month"],
            subset["Rating"],
            marker="o",
            label=b
        )

    plt.title("Average Rating Over Time by Branch")
    plt.xlabel("Year-Month")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45)

    plt.legend(title="Branch")
    plt.tight_layout()

    # 4. 保存并展示
    plt.savefig("rating_trend_by_branch.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
