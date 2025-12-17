import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from .formatters import millions_formatter

def bar(df, category_col, value_col, out: Path):
    agg = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)

    plt.figure(figsize=(7, 4))
    bars = plt.bar(agg.index, agg.values)

    plt.title(
        f"{agg.index[0]} Drives {agg.iloc[0]/agg.sum():.1%} of Total Revenue",
        fontsize=11,
        weight="bold"
    )
    plt.ylabel("Revenue")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
