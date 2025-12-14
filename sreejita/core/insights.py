import numpy as np

def InsightGenerator(df, target="sales"):
    insights = []
    num = df.select_dtypes(include=[np.number])

    if target not in num.columns:
        return insights

    corr = num.corr()
    strongest = corr[target].drop(target).abs().sort_values(ascending=False)

    if not strongest.empty:
        col = strongest.index[0]
        val = corr.loc[target, col]
        insights.append(f"{target.title()} strongly correlates with {col} (r={val:.2f}).")

    return insights
