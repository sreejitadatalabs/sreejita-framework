def compute_ops_kpis(df):
    return {
        "avg_process_time": df["process_time"].mean(),
        "sla_breach_rate": df["sla_breached"].mean()
    }
