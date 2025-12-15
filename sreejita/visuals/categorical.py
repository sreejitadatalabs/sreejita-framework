import seaborn as sns
import matplotlib.pyplot as plt
from sreejita.core.schema import detect_schema


def bar(df, col, out):
    schema = detect_schema(df)

    if col not in schema["categorical"]:
        return

    data = df[col].value_counts().head(8)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=data.values, y=data.index, ax=ax)
    ax.set_title(f"Top Categories — {col}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
