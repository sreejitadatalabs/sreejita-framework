# =====================================================
# COLUMN NORMALIZER — STABLE & DOMAIN-SAFE
# Sreejita Framework v3.6
# =====================================================

import re
from typing import Iterable, Tuple, Dict, Set


# -------------------------------------------------
# SINGLE COLUMN NORMALIZATION
# -------------------------------------------------

def normalize_column(col: str) -> str:
    """
    Normalize column names into stable snake_case tokens.

    GUARANTEES:
    - Preserves semantic word boundaries
    - Never returns empty string
    - Never raises
    """

    try:
        if col is None:
            return ""

        col = str(col).strip().lower()

        # Replace separators with space (preserve word boundaries)
        col = re.sub(r"[\/\-\.\(\)\[\]%]", " ", col)

        # Remove remaining non-alphanumerics
        col = re.sub(r"[^a-z0-9\s_]", "", col)

        # Normalize whitespace to underscores
        col = re.sub(r"\s+", "_", col)

        # Collapse multiple underscores
        col = re.sub(r"_+", "_", col)

        # Strip edge underscores
        col = col.strip("_")

        return col or ""

    except Exception:
        return ""


# -------------------------------------------------
# BULK NORMALIZATION
# -------------------------------------------------

def normalize_columns(
    columns: Iterable[str],
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Normalize iterable of column names.

    Returns:
      - normalized_columns: set[str]
      - mapping: original_column -> normalized_column
    """

    mapping: Dict[str, str] = {}

    for c in columns:
        normalized = normalize_column(c)
        if normalized:
            mapping[c] = normalized

    return set(mapping.values()), mapping
