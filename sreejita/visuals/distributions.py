import seaborn as sns
import matplotlib.pyplot as plt

def hist(df, col, out):
    fig, ax = plt.subplots(figsize=(5,3))
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
