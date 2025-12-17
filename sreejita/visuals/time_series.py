from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def human_currency(x, _):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    else:
        return f"${x:,.0f}"


def sales_trend(df, output_path: Path, sales_col="sales"):
    date_candidates = [
        "order_date", "Order Date", "date", "Date", "transaction_date"
    ]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if not date_col or sales_col not in df.columns:
        return None

    data = df[[date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    if data.empty:
        return None

    monthly = (
        data
        .set_index(date_col)
        .resample("M")[sales_col]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2)

    ax.set_title("Sales Show a Stable Upward Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")

    # 🔥 FIX: remove scientific notation
    ax.ticklabel_format(style="plain", axis="y")

    # 🔥 FIX: human-readable axis
    ax.yaxis.set_major_formatter(FuncFormatter(human_currency))

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
