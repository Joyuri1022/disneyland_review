import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("review_clean.csv")

    branch_counts = df["Branch"].value_counts()

    plt.figure(figsize=(10, 6))
    branch_counts.plot(kind="bar")

    plt.title("Disneyland Branch Distribution")
    plt.xlabel("Branch")
    plt.ylabel("Count")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("branch_distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
