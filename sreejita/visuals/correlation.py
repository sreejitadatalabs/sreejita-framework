import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from .formatters import thousands_formatter

def shipping_vs_sales(df, sales_col, ship_col, category_col, out: Path):
    plt.figure(figsize=(7, 4))

    for cat in df[category_col].unique():
        subset = df[df[category_col] == cat]
        plt.scatter(
            subset[sales_col],
            subset[ship_col],
            alpha=0.4,
            label=cat
        )

    plt.title(
        "Shipping Cost Increases with Sales Volume (Category-Specific Patterns)",
        fontsize=11,
        weight="bold"
    )
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
