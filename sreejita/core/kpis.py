def KPICalculator(df, base_kpis=None, domain_kpis=None):
    kpis = {
        "Rows": df.shape[0],
        "Columns": df.shape[1]
    }

    if base_kpis:
        kpis.update(base_kpis)

    if domain_kpis:
        kpis.update(domain_kpis)

    return kpis
