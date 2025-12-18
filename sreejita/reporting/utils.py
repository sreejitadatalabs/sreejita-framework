def safe_mean(df, column):
    return df[column].mean() if column in df.columns else None


def safe_sum(df, column):
    return df[column].sum() if column in df.columns else None


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator
