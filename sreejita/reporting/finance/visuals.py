from pathlib import Path
import matplotlib.pyplot as plt


# ---------- PUBLIC API ----------

def expense_vs_revenue(df, output_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(df["revenue"], label="Revenue")
    plt.plot(df["expenses"], label="Expenses")
    plt.legend()
    plt.title("Revenue vs Expenses")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# ---------- INTERNAL v2.6 ----------

def _expense_vs_revenue_v26(df, output_dir: Path):
    path = output_dir / "expense_vs_revenue.png"
    return expense_vs_revenue(df, path)
