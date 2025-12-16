from pathlib import Path
import matplotlib.pyplot as plt


# ---------- PUBLIC API ----------

def sla_breach_rate_plot(df, output_path: Path):
    plt.figure(figsize=(6, 4))
    df["sla_breached"].value_counts().plot(kind="bar")
    plt.title("SLA Breach Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# ---------- INTERNAL v2.6 ----------

def _sla_breach_rate_plot_v26(df, output_dir: Path):
    path = output_dir / "sla_breach_rate.png"
    return sla_breach_rate_plot(df, path)
