def compute_healthcare_kpis(df):
    return {
        "avg_outcome_score": df["outcome_score"].mean(),
        "readmission_rate": (df["readmitted"] == True).mean(),
        "patient_count": len(df),
    }
