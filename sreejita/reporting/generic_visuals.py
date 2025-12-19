import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Columns that should NEVER be visualized
EXCLUDE_COL_KEYWORDS = {
    "id", "uuid", "row", "index", "order_id",
    "customer_id", "patient_id"
}


def _is_meaningful(col: str) -> bool:
    name = col.lower()
    return not any(k in name for k in EXCLUDE_COL_KEYWORDS)


def generate_generic_visuals(df: pd.DataFrame, output_dir: Path, max_visuals=2):
    """
    Generic visuals are LAST RESORT and BUSINESS-SAFE.
    Never visualize IDs or keys.
    """
    visuals = []
    output_dir.mkdir(exist_ok=True, parents=True)

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if _is_meaningful(c)
    ]

    cat_cols = [
        c for c in df.select_dtypes(exclude="number").columns
        if _is_meaningful(c)
    ]

    # 1️⃣ Numeric distribution (only meaningful)
    if numeric_cols and len(visuals) < max_visuals:
        col = numeric_cols[0]
        path = output_dir / f"{col}_distribution.png"
        df[col].hist(figsize=(6, 4))
        plt.title(f"{col.replace('_',' ').title()} Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": f"Distribution of {col.replace('_',' ')}"
        })

    # 2️⃣ Missing values overview (only if missing exists)
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty and len(visuals) < max_visuals:
        path = output_dir / "missing_values.png"
        missing.plot(kind="bar", figsize=(8, 4))
        plt.title("Missing Values by Column")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Data completeness issues"
        })

    return visuals
