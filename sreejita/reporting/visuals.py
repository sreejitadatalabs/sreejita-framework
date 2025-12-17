import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path


def human_readable_number(x, _):
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{int(x)}"


def shipping_cost_vs_sales(df, output_dir: Path):
    path = output_dir / "shipping_cost_vs_sales.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(df["sales"], df["shipping_cost"], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax.ticklabel_format(style="plain", axis="both")

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
