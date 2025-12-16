from pathlib import Path
import matplotlib.pyplot as plt


def shipping_cost_vs_sales(df, output_path: Path):
    """
    PUBLIC API (DO NOT BREAK)
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(df["sales"], df["shipping_cost"], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
