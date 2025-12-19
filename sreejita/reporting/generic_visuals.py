import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def generate_generic_visuals(df: pd.DataFrame, output_dir: Path, max_visuals=4):
    visuals = []
    output_dir.mkdir(exist_ok=True, parents=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # 1️⃣ Numeric distribution
    if numeric_cols and len(visuals) < max_visuals:
        col = numeric_cols[0]
        path = output_dir / f"{col}_distribution.png"
        df[col].hist(figsize=(6, 4))
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": f"Distribution of {col}"
        })

    # 2️⃣ Missing values
    if len(visuals) < max_visuals:
        path = output_dir / "missing_values.png"
        df.isna().sum().plot(kind="bar", figsize=(8, 4))
        plt.title("Missing Values by Column")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Data completeness overview"
        })

    # 3️⃣ Top categories
    if cat_cols and len(visuals) < max_visuals:
        col = cat_cols[0]
        path = output_dir / f"{col}_top_categories.png"
        df[col].value_counts().head(10).plot(kind="bar", figsize=(6, 4))
        plt.title(f"Top categories in {col}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": f"Top categories in {col}"
        })

    # 4️⃣ Record count
    if len(visuals) < max_visuals:
        path = output_dir / "record_count.png"
        pd.Series({"Records": len(df)}).plot(kind="bar", figsize=(4, 4))
        plt.title("Record Volume")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Total record volume"
        })

    return visuals[:max_visuals]
