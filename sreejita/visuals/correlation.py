import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sreejita.core.schema import detect_schema


def heatmap(df, out):
    """
    Correlation heatmap for numeric features.
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

    ax.set_title("Key numeric metrics show varying degrees of correlation")

    fig.tight_layout()
    fig.savefig(str(out), dpi=300)
    plt.close(fig)

    if out.exists() and out.stat().st_size > 0:
        return out

    return None


def shipping_cost_vs_sales(df, sales_col, shipping_col, out):
    """
    Scatter plot with insight-driven title.
    """

    schema = detect_schema(df)

    if sales_col not in schema["numeric_measures"]:
        return

    if shipping_col not in schema["numeric_measures"]:
        return

    corr = df[shipping_col].corr(df[sales_col])
    direction = "increases with" if corr > 0 else "decreases with"

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.scatterplot(
        x=df[sales_col],
        y=df[shipping_col],
        alpha=0.4,
        ax=ax
    )

    ax.set_title(
        f"Shipping cost {direction} order value across transactions"
    )

    ax.set_xlabel("Order value")
    ax.set_ylabel("Shipping cost")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close()
