import hashlib

def dataframe_fingerprint(df) -> str:
    content = (
        str(df.columns.tolist()) +
        str(df.dtypes.tolist()) +
        str(df.head(50).to_dict())
    )
    return hashlib.sha256(content.encode()).hexdigest()
