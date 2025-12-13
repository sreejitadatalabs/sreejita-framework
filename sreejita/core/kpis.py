def compute_kpis(df, base_metrics=None):
    kpis = {
        "Rows": df.shape[0],
        "Columns": df.shape[1]
    }

    if base_metrics:
        for name, col in base_metrics.items():
            if col in df.columns:
                kpis[name] = round(df[col].sum(), 2)

    return kpis
