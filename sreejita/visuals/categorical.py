import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.schema import detect_schema
from sreejita.reporting.formatters import fmt_currency


def bar(df, col, out):
    """
    Category bar chart with executive-grade visual polish.
    Insight-driven, deterministic, PDF-readable.
    """

    schema = detect_schema(df)

    if col not in schema["categorical"]:
        return

    if not schema["numeric_measures"]:
        return

    # Use first numeric measure (usually sales / revenue)
    value_col = schema["numeric_measures"][0]

    data = (
        df.groupby(col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )

    if data.empty:
        return

    top_category = data.index[0]
    top_share = (data.iloc[0] / data.sum()) * 100

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.barplot(
        x=data.values,
        y=data.index,
        ax=ax,
        palette="Blues_r",
    )

    # -------------------------------
    # INSIGHT-DRIVEN TITLE (D1)
    # -------------------------------
    ax.set_title(
        f"{top_category} contributes the largest share of total value ({top_share:.1f}%)",
        fontsize=11,
        pad=10,
    )

    # -------------------------------
    # SEMANTIC AXIS LABELS (D3)
    # -------------------------------
    ax.set_xlabel("Total Order Value ($)")
    ax.set_ylabel(col.replace("_", " ").title())

    # -------------------------------
    # FORMATTING & READABILITY (D2, D4)
    # -------------------------------
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, _: fmt_currency(x))
    )

    # Disable scientific notation defensively
    ax.ticklabel_format(style="plain", axis="x")
    ax.ticklabel_format(style="plain", axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
