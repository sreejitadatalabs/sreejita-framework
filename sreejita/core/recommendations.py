def generate_recommendations(df, sales="sales", profit="profit"):
    recs = []

    if sales in df.columns and profit in df.columns:
        margin = df[profit].sum() / df[sales].sum()
        if margin < 0.1:
            recs.append("Low profit margin detected. Review pricing and discount strategy.")
        else:
            recs.append("Healthy profit margin. Focus on scaling top-performing segments.")

    if "discount" in df.columns:
        recs.append("Evaluate discount effectiveness to reduce margin leakage.")

    return recs
