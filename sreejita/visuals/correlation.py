import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sreejita.core.schema import detect_schema


def heatmap(df, out):
    """
    Guaranteed-safe heatmap renderer.
    Saves a non-empty PNG usable by ReportLab.
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
    ax.set_title("Correlation Matrix")

    fig.tight_layout()
    fig.savefig(str(out), dpi=300)
    plt.close(fig)

    # 🔒 HARD GUARANTEE
    if out.exists() and out.stat().st_size > 0:
        return out

    return None
