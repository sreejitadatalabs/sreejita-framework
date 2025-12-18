import pandas as pd

def safe_mean(df, column):
    if column in df.columns:
        try:
            return pd.to_numeric(df[column], errors='coerce').mean()
        except:
            return None
    return None

def safe_sum(df, column):
    if column in df.columns:
        try:
            return pd.to_numeric(df[column], errors='coerce').sum()
        except:
            return None
    return None

def safe_ratio(numerator, denominator):
    try:
        numerator = pd.to_numeric(numerator, errors='coerce')
        denominator = pd.to_numeric(denominator, errors='coerce')
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator
    except:
        return None
