def generate_recommendations(df, sales="sales", profit="profit"):
    recs = []

    if sales in df.columns and profit in df.columns:
        margin = df[profit].sum() / df[sales].sum()
        if margin < 0.1:
            recs.append(
                "Low profit margin detected. Review pricing and discount strategy to reduce margin erosion risk."
            )
        else:
            recs.append(
                "Healthy profit margin observed. Focus on scaling top-performing segments and channels."
            )

    if "discount" in df.columns:
        high_disc = (df["discount"] > 0.3).mean() * 100
        recs.append(
            f"Approximately {high_disc:.1f}% of transactions use high discounts. "
            f"Consider introducing discount caps or approval thresholds."
        )

    return recs
