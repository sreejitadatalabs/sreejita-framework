def compute_healthcare_kpis(df):
    return {
        "avg_wait_time": df["wait_time"].mean(),
        "readmission_rate": df["readmitted"].mean()
    }
