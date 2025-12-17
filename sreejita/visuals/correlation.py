import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

from sreejita.core.schema import detect_schema
from sreejita.reporting.formatters import fmt_currency


def heatmap(df, out):
    """
    Correlation heatmap for numeric features.
    Executive-grade visual polish applied.
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

    # -------------------------------
    # INSIGHT-DRIVEN TITLE (D1)
    # -------------------------------
    ax.set_title(
        "Key numeric metrics exhibit meaningful correlation patterns",
        fontsize=11,
        pad=10,
    )

    fig.tight_layout()
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if out.exists() and out.stat().st_size > 0:
        return out

    return None


def shipping_cost_vs_sales(df, sales_col, shipping_col, out):
    """
    Scatter plot with executive-grade visual polish.
    Insight-driven, deterministic.
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
        alpha=0.4,      # D4: density visibility
        s=30,
        ax=ax,
    )

    # -------------------------------
    # INSIGHT-DRIVEN TITLE (D1)
    # -------------------------------
    ax.set_title(
        f"Shipping cost {direction} order value across transactions",
        fontsize=11,
        pad=10,
    )

    # -------------------------------
    # SEMANTIC AXIS LABELS (D3)
    # -------------------------------
    ax.set_xlabel("Order Value ($)")
    ax.set_ylabel("Shipping Cost ($)")

    # -------------------------------
    # FORMATTING & READABILITY (D2, D4)
    # -------------------------------
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, _: fmt_currency(x))
    )
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda y, _: fmt_currency(y))
    )

    # Disable scientific notation defensively
    ax.ticklabel_format(style="plain", axis="x")
    ax.ticklabel_format(style="plain", axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
