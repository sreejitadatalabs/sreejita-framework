def compute_healthcare_kpis(df):
    return {
        "outcome_score": df["outcome_score"].mean(),
        "readmission_rate": df["readmitted"].mean(),
    }
