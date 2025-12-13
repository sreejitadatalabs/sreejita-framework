import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(df, out):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
