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

from pathlib import Path
import matplotlib.pyplot as plt


# =====================================================
# PUBLIC VISUALS (CI SAFE)
# =====================================================

def sales_trend(df, output_path: Path,
                date_col="order_date",
                sales_col="sales"):
    """
    WHAT happened — Monthly sales trend
    """
    if date_col not in df.columns or sales_col not in df.columns:
        return None

    data = df[[date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    if data.empty:
        return None

    monthly = (
        data
        .set_index(date_col)
        .resample("M")[sales_col]
        .sum()
    )

    plt.figure(figsize=(7, 4))
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.title("Sales Trend Over Time")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def sales_by_category(df, output_path: Path, category_col="category", sales_col="sales"):
    """
    WHERE it happened — sales concentration
    """
    if category_col not in df.columns:
        return None

    agg = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    agg.plot(kind="bar")
    plt.title("Sales by Category")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def shipping_cost_vs_sales(df, output_path: Path):
    """
    WHY it happened — cost efficiency
    """
    if not {"sales", "shipping_cost"}.issubset(df.columns):
        return None

    plt.figure(figsize=(6, 4))
    plt.scatter(df["sales"], df["shipping_cost"], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# =====================================================
# INTERNAL v2.7 HELPERS (USED BY REGISTRY)
# =====================================================

def _sales_trend_v27(df, output_dir: Path):
    return sales_trend(df, output_dir / "sales_trend.png")


def _sales_by_category_v27(df, output_dir: Path):
    return sales_by_category(df, output_dir / "sales_by_category.png")


def _shipping_cost_vs_sales_v27(df, output_dir: Path):
    return shipping_cost_vs_sales(df, output_dir / "shipping_cost_vs_sales.png")


def _shipping_cost_vs_sales_v26(df, output_dir: Path):
    return shipping_cost_vs_sales(df, output_dir / "shipping_cost_vs_sales.png")


def _discount_distribution_v26(df, output_dir: Path):
    return discount_distribution(df, output_dir / "discount_distribution.png")


def _baseline_sales_distribution_v27(df, output_dir: Path):
    return baseline_sales_distribution(
        df, output_dir / "baseline_sales_distribution.png"
    )
