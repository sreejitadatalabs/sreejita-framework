import numpy as np
from sreejita.core.schema import split_column_roles

def correlation_insights(df, target="sales"):
    insights = []
    
    roles = split_column_roles(df)
    num_cols = roles["numeric_measures"]

    if target not in num.columns:
        return insights

    corr = num.corr()
    strongest = corr[target].drop(target).abs().sort_values(ascending=False)

    if not strongest.empty:
        col = strongest.index[0]
        val = corr.loc[target, col]
        insights.append(f"{target.title()} strongly correlates with {col} (r={val:.2f}).")

    return insights
