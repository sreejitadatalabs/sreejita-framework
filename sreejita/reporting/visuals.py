import matplotlib.pyplot as plt
from pathlib import Path


def shipping_cost_vs_sales(df, output_dir: Path):
    path = output_dir / "shipping_cost_vs_sales.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(df["sales"], df["shipping_cost"], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def discount_distribution(df, output_dir: Path):
    path = output_dir / "discount_distribution.png"
    plt.figure(figsize=(6, 4))
    plt.hist(df["discount"], bins=20, alpha=0.7)
    plt.xlabel("Discount Rate")
    plt.ylabel("Frequency")
    plt.title("Distribution of Discounts")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
