def enrich(df):
    if "sentiment_score" in df.columns:
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda x: "positive" if x > 0.2 else "negative"
        )
    return df
