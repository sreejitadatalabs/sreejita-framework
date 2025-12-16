from pathlib import Path
import matplotlib.pyplot as plt


# =====================================================
# PUBLIC API (CI-SAFE)
# =====================================================

def shipping_cost_vs_sales(df, output_path: Path):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["sales"], df["shipping_cost"], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def discount_distribution(df, output_path: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(df["discount"], bins=20, alpha=0.7)
    plt.xlabel("Discount Rate")
    plt.ylabel("Frequency")
    plt.title("Discount Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def baseline_sales_distribution(df, output_path: Path):
    """
    BASELINE visual — ALWAYS SAFE
    """
    plt.figure(figsize=(6, 4))
    plt.hist(df["sales"], bins=30, alpha=0.7)
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.title("Sales Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# =====================================================
# INTERNAL v2.6 / v2.7 HELPERS
# =====================================================

def _shipping_cost_vs_sales_v26(df, output_dir: Path):
    return shipping_cost_vs_sales(df, output_dir / "shipping_cost_vs_sales.png")


def _discount_distribution_v26(df, output_dir: Path):
    return discount_distribution(df, output_dir / "discount_distribution.png")


def _baseline_sales_distribution_v27(df, output_dir: Path):
    return baseline_sales_distribution(
        df, output_dir / "baseline_sales_distribution.png"
    )
