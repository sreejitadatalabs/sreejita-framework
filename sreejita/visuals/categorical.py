import seaborn as sns
import matplotlib.pyplot as plt

def bar(df, col, out):
    data = df[col].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=data.values, y=data.index, ax=ax)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
