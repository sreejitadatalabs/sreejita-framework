import matplotlib.pyplot as plt
import pandas as pd

def plot_monthly(df, date_col, value_col, out):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    m = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum()
    m.index = m.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(m.index, m.values)
    ax.set_title("Monthly Trend")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
