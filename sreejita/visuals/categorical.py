import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

def _m_formatter(x, _):
    return f"${x/1_000_000:.1f}M"

def category_sales_visual(df, output_dir: Path):
    cat_col = next(
        (c for c in df.columns if "category" in c.lower()),
        None
    )
    sales_col = next(
        (c for c in df.columns if "sales" in c.lower()),
        None
    )

    if not cat_col or not sales_col:
        return None

    agg = (
        df.groupby(cat_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
    )

    if agg.empty:
        return None

    out = output_dir / "category_sales.png"

    plt.figure(figsize=(7, 4))
    agg.plot(kind="bar", color="#4C72B0")

    plt.title(
        f"{agg.index[0]} Drives {agg.iloc[0]/agg.sum():.0%} of Revenue",
        weight="bold"
    )
    plt.ylabel("Revenue")

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(_m_formatter))
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()

    return out
