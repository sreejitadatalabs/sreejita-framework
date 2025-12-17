import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

def _k_formatter(x, _):
    return f"${x/1_000:.0f}K"

def sales_trend_visual(df, output_dir: Path):
    date_col = next(
        (c for c in df.columns if "date" in c.lower()),
        None
    )
    sales_col = next(
        (c for c in df.columns if "sales" in c.lower()),
        None
    )

    if not date_col or not sales_col:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    monthly = (
        df.dropna(subset=[date_col])
        .groupby(df[date_col].dt.to_period("M"))[sales_col]
        .sum()
    )

    if monthly.empty:
        return None

    out = output_dir / "sales_trend.png"

    plt.figure(figsize=(7, 4))
    plt.plot(monthly.index.to_timestamp(), monthly.values, marker="o")
    plt.title("Sales Trend Shows Stable Monthly Performance", weight="bold")
    plt.ylabel("Monthly Sales")

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out
