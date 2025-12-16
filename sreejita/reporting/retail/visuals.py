from pathlib import Path
import matplotlib.pyplot as plt


# =====================================================
# PUBLIC API — REQUIRED BY CI / CLI / SCHEDULER
# =====================================================

def shipping_cost_vs_sales(df, output_path: Path):
    """
    PUBLIC API (legacy + required by CI)
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


def discount_distribution(df, output_path: Path):
    """
    PUBLIC API — required by CI
    """
    plt.figure(figsize=(6, 4))
    plt.hist(df["discount"], bins=20, alpha=0.7)
    plt.xlabel("Discount Rate")
    plt.ylabel("Frequency")
    plt.title("Discount Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# =====================================================
# INTERNAL v2.6 HELPERS (SAFE TO EVOLVE)
# =====================================================

def _shipping_cost_vs_sales_v26(df, output_dir: Path):
    """
    INTERNAL helper for v2.6 insight-driven visuals
    """
    output_path = output_dir / "shipping_cost_vs_sales.png"
    return shipping_cost_vs_sales(df, output_path)


def _discount_distribution_v26(df, output_dir: Path):
    """
    INTERNAL helper for v2.6 insight-driven visuals
    """
    output_path = output_dir / "discount_distribution.png"
    return discount_distribution(df, output_path)
