from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def human_readable_number(x, _):
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{int(x)}"


# =====================================================
# WHAT happened — Sales Trend
# =====================================================
def sales_trend(df, output_path: Path, sales_col="sales"):
    date_candidates = [
        "order_date", "Order Date", "date", "Date", "transaction_date"
    ]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if not date_col or sales_col not in df.columns:
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

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


# =====================================================
# WHERE it happened — Sales by Category
# =====================================================
def sales_by_category(df, output_path: Path,
                      category_col="category", sales_col="sales"):
    if category_col not in df.columns or sales_col not in df.columns:
        return None

    agg = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    agg.plot(kind="bar")
    plt.title("Sales by Category")
    plt.ylabel("Sales")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


# =====================================================
# WHY it happened — Shipping Cost vs Sales
# =====================================================
def shipping_cost_vs_sales(df, output_path: Path):
    if not {"sales", "shipping_cost"}.issubset(df.columns):
        return None

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
    plt.savefig(output_path)
    plt.close()

    return output_path
