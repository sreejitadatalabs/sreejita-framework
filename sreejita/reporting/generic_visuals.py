import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def generate_generic_visuals(df: pd.DataFrame, output_dir: Path):
    visuals = []
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1️⃣ Missing values heatmap (simplified bar)
    missing = df.isna().sum()
    if not missing.empty:
        path = output_dir / "missing_values.png"
        missing.plot(kind="bar", figsize=(8, 4))
        plt.title("Missing Values by Column")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({"path": path, "caption": "Missing values across dataset columns"})

    # 2️⃣ Numeric distribution
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        col = numeric.columns[0]
        path = output_dir / "distribution.png"
        numeric[col].hist(figsize=(6, 4))
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({"path": path, "caption": f"Distribution of {col}"})

    # 3️⃣ Top categories
    categorical = df.select_dtypes(include="object")
    if not categorical.empty:
        col = categorical.columns[0]
        path = output_dir / "top_categories.png"
        df[col].value_counts().head(10).plot(kind="bar", figsize=(6, 4))
        plt.title(f"Top categories in {col}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({"path": path, "caption": f"Top categories in {col}"})

    return visuals[:4]
