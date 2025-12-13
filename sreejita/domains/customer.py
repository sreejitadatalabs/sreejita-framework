from sklearn.cluster import KMeans

def enrich(df):
    if "spend" in df.columns:
        df["segment"] = KMeans(n_clusters=3).fit_predict(df[["spend"]])
    return df
