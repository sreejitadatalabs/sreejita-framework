from pathlib import Path
import pandas as pd

from sreejita.core.schema import detect_schema
from sreejita.reporting.formatters import fmt_currency
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


def _sales_trend_v27(df, output_dir: Path, config: dict):
    """
    Sales trend over time (v2.7+ hook-compatible)
    """

    dataset_cfg = config.get("dataset", {})
    date_col = dataset_cfg.get("date")
    sales_col = dataset_cfg.get("sales")

    if not date_col or not sales_col:
        return None

    schema = detect_schema(df)
    if date_col not in schema["datetime"]:
        return None
    if sales_col not in schema["numeric_measures"]:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    monthly = df.groupby(df[date_col].dt.to_period("M"))[sales_col].sum()
    if monthly.empty:
        return None

    monthly.index = monthly.index.to_timestamp()

    out = output_dir / "sales_trend.png"

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2)

    trend = "Upward" if monthly.values[-1] > monthly.values[0] else "Flat"
    ax.set_title(f"Sales Trend is {trend} Over the Observed Period")

    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, _: fmt_currency(x))
    )
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.autofmt_xdate()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    return str(out)
