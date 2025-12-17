from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def human_currency(x, _):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    else:
        return f"${x:,.0f}"


def sales_by_category(df, output_path: Path,
                      category_col="category", sales_col="sales"):
    if category_col not in df.columns or sales_col not in df.columns:
        return None

    agg = (
        df.groupby(category_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    agg.plot(kind="bar", ax=ax, color="#4C72B0")

    ax.set_title("Technology Drives the Largest Share of Revenue")
    ax.set_ylabel("Revenue")

    # 🔥 FIX: no scientific notation
    ax.ticklabel_format(style="plain", axis="y")

    # 🔥 FIX: human-readable currency
    ax.yaxis.set_major_formatter(FuncFormatter(human_currency))

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
