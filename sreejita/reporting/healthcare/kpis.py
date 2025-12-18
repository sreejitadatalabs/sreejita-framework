from sreejita.reporting.utils import safe_mean, safe_sum


def compute_healthcare_kpis(df):
    return {
        "avg_billing_amount": safe_mean(df, "billing_amount"),
        "avg_length_of_stay": safe_mean(df, "length_of_stay"),
        "readmission_rate": safe_mean(df, "readmitted"),
        "avg_outcome_score": safe_mean(df, "outcome_score"),  # optional
        "total_patients": safe_sum(df, "patient_id"),
    }
