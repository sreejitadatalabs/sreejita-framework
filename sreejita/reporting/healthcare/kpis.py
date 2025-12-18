def compute_healthcare_kpis(df):
    """
    Compute healthcare KPIs defensively.
    Missing columns must not crash reporting.
    """

    kpis = {}

    if "billing_amount" in df.columns:
        kpis["avg_billing_amount"] = df["billing_amount"].mean()
    else:
        kpis["avg_billing_amount"] = None

    if "length_of_stay" in df.columns:
        kpis["avg_length_of_stay"] = df["length_of_stay"].mean()
    else:
        kpis["avg_length_of_stay"] = None

    if "readmitted" in df.columns:
        kpis["readmission_rate"] = df["readmitted"].mean()
    else:
        kpis["readmission_rate"] = None

    # OPTIONAL / advanced KPI
    if "outcome_score" in df.columns:
        kpis["avg_outcome_score"] = df["outcome_score"].mean()
    else:
        kpis["avg_outcome_score"] = None

    return kpis
