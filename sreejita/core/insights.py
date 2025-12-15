import numpy as np
from sreejita.core.schema import detect_schema


def correlation_insights(df, target="sales"):
    insights = []

    schema = detect_schema(df)
    num_cols = schema["numeric_measures"]

    if target not in num_cols or len(num_cols) < 2:
        return insights

    corr = df[num_cols].corr()
    strongest = corr[target].drop(target).abs().sort_values(ascending=False)

    if not strongest.empty:
        col = strongest.index[0]
        val = corr.loc[target, col]
        insights.append(
            f"{target.title()} strongly correlates with {col} (r={val:.2f})."
        )

    return insights
