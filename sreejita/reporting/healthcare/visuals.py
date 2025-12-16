from pathlib import Path
import matplotlib.pyplot as plt


# ---------- PUBLIC API ----------

def readmission_rate_plot(df, output_path: Path):
    plt.figure(figsize=(6, 4))
    df["readmitted"].value_counts().plot(kind="bar")
    plt.title("Patient Readmission Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# ---------- INTERNAL v2.6 ----------

def _readmission_rate_plot_v26(df, output_dir: Path):
    path = output_dir / "readmission_rate.png"
    return readmission_rate_plot(df, path)
