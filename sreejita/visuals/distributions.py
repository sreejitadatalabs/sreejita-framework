import seaborn as sns
import matplotlib.pyplot as plt
from sreejita.core.schema import detect_schema


def hist(df, col, out):
    schema = detect_schema(df)

    # Only allow true numeric measures
    if col not in schema["numeric_measures"]:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution — {col}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
