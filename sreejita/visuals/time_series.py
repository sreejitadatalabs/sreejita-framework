import matplotlib.pyplot as plt
import pandas as pd
from sreejita.core.schema import detect_schema


def plot_monthly(df, date_col, value_col, out):
    schema = detect_schema(df)

    if date_col not in schema["datetime"]:
        return

    if value_col not in schema["numeric_measures"]:
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    m = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum()
    m.index = m.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(m.index, m.values)
    ax.set_title("Monthly Trend")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
