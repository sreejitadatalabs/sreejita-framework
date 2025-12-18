import re

def normalize_column(col: str) -> str:
    """
    Normalize column names into canonical snake_case tokens.
    Examples:
      'Admission Date' -> 'admission_date'
      'Shipping Cost' -> 'shipping_cost'
      'CTR (%)' -> 'ctr'
    """
    col = col.lower().strip()
    col = re.sub(r"[^\w\s]", "", col)   # remove symbols
    col = re.sub(r"\s+", "_", col)      # spaces to underscores
    return col


def normalize_columns(columns):
    """
    Returns:
      - normalized_columns: set[str]
      - mapping: original -> normalized
    """
    mapping = {c: normalize_column(c) for c in columns}
    return set(mapping.values()), mapping
