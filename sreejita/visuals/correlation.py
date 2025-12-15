import seaborn as sns
import matplotlib.pyplot as plt
from sreejita.core.schema import detect_schema


def heatmap(df, out):
    schema = detect_schema(df)
    num_cols = schema["numeric_measures"]

    if len(num_cols) < 2:
        return

    corr_df = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
