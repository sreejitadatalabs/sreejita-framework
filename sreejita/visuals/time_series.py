import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from .formatters import thousands_formatter

def plot_monthly(df, date_col, value_col, out: Path):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    monthly = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum()
    monthly.index = monthly.index.to_timestamp()

    plt.figure(figsize=(7, 4))
    plt.plot(monthly.index, monthly.values, marker="o")

    growth = (monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0]

    plt.title(
        f"Sales Trending {'UP' if growth > 0 else 'DOWN'} ({growth:.1%} change)",
        fontsize=11,
        weight="bold"
    )
    plt.ylabel("Monthly Sales")
    plt.xlabel("Month")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
