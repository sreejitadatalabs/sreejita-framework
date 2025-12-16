from pathlib import Path
import matplotlib.pyplot as plt


# ---------- PUBLIC API ----------

def churn_proxy_distribution(df, output_path: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(df["last_purchase_days"], bins=30, alpha=0.7)
    plt.xlabel("Days Since Last Purchase")
    plt.ylabel("Customers")
    plt.title("Customer Inactivity Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# ---------- INTERNAL v2.6 ----------

def _churn_proxy_distribution_v26(df, output_dir: Path):
    path = output_dir / "customer_churn_proxy.png"
    return churn_proxy_distribution(df, path)
