import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from sreejita.core.schema import detect_schema
from sreejita.reporting.formatters import fmt_currency


def plot_monthly(df, date_col, value_col, out):
    """
    Monthly trend plot with insight-driven title.
    Deterministic, no thresholds.
    """

    schema = detect_schema(df)

    if date_col not in schema["datetime"]:
        return

    if value_col not in schema["numeric_measures"]:
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    monthly = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum()
    monthly.index = monthly.index.to_timestamp()

    if monthly.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
        linewidth=2,
    )

    # ---------- INSIGHT-DRIVEN TITLE ----------
    trend = "upward" if monthly.values[-1] > monthly.values[0] else "flat or declining"

    ax.set_title(
        f"Overall performance shows a {trend} trend over the observed period"
    )
    # -----------------------------------------

    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, _: fmt_currency(x))
    )
    ax.ticklabel_format(style="plain", axis="y")

    fig.autofmt_xdate()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
