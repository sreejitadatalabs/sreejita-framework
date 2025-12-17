import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

def _k_formatter(x, _):
    return f"${x/1_000:.0f}K"

def shipping_vs_sales_visual(df, output_dir: Path):
    sales_col = next((c for c in df.columns if "sales" in c.lower()), None)
    ship_col = next((c for c in df.columns if "ship" in c.lower()), None)
    cat_col = next((c for c in df.columns if "category" in c.lower()), None)

    if not sales_col or not ship_col or not cat_col:
        return None

    out = output_dir / "shipping_vs_sales.png"

    plt.figure(figsize=(7, 4))
    for cat in df[cat_col].dropna().unique():
        subset = df[df[cat_col] == cat]
        plt.scatter(
            subset[sales_col],
            subset[ship_col],
            alpha=0.4,
            label=cat
        )

    plt.title(
        "Shipping Cost Rises with Sales Volume (Category Patterns)",
        weight="bold"
    )
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="both")
    ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
    plt.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()

    return out
