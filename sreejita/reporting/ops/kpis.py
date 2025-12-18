def compute_ops_kpis(df):
    return {
        "avg_cycle_time": df["cycle_time"].mean(),
        "on_time_rate": (df["on_time"] == True).mean(),
        "records_processed": len(df),
    }
