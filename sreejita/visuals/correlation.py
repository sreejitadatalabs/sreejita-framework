import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sreejita.core.schema import detect_schema


def heatmap(df, out):
    """
    Guaranteed-safe heatmap renderer.
    Saves a non-empty PNG usable by ReportLab.
    """
    schema = detect_schema(df)
    num_cols = schema["numeric_measures"]

    if len(num_cols) < 2:
        return None

    corr_df = df[num_cols].corr()

    out = Path(out).resolve()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    ax.set_title("Correlation Matrix")

    fig.tight_layout()
    fig.savefig(str(out), dpi=300)
    plt.close(fig)

    # 🔒 HARD GUARANTEE
    if out.exists() and out.stat().st_size > 0:
        return out

    return None

def shipping_cost_vs_sales(df, sales_col, shipping_col, out):
    """
    Scatter plot: Shipping Cost vs Sales
    Visual polish only (v2.8.2)
    """

    schema = detect_schema(df)

    if sales_col not in schema["numeric_measures"]:
        return
    if shipping_col not in schema["numeric_measures"]:
        return

    # Optional category coloring
    category_col = schema["categorical"][0] if schema["categorical"] else None

    plt.figure(figsize=(6, 4))

    sns.scatterplot(
        data=df,
        x=sales_col,
        y=shipping_col,
        hue=category_col,
        alpha=0.4,      # transparency to show density
        s=35,           # marker size
        edgecolor=None,
    )

    plt.title("Shipping Cost Efficiency Varies by Product Category")
    plt.xlabel("Sales Value")
    plt.ylabel("Shipping Cost")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
