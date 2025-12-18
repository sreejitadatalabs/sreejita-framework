from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio
from sreejita.domains.column_mapping import ColumnMapping

def compute_healthcare_kpis(df):
    """
    Compute healthcare KPIs with flexible column matching.
    Supports multiple naming conventions for clinical and patient metrics.
    """
    # Auto-detect and map columns
    mapping = ColumnMapping.auto_detect(df)
    
    # Define column alternatives for flexible matching
    outcome_score_cols = ['outcome_score', 'score', 'health_score', 'clinical_outcome', 'clinical_score']
    readmitted_cols = ['readmitted', 'readmission', 'rehospitalized', 'readmission_flag', 'readmit_flag']
    length_of_stay_cols = ['length_of_stay', 'los', 'days_admitted', 'stay_duration', 'admission_days']
    patient_id_cols = ['patient_id', 'id', 'encounter_id', 'medical_record_id', 'mrn']
    
    # Find the actual columns in the dataframe
    outcome_col = next((col for col in outcome_score_cols if col in df.columns), None)
    readmit_col = next((col for col in readmitted_cols if col in df.columns), None)
    los_col = next((col for col in length_of_stay_cols if col in df.columns), None)
    patient_col = next((col for col in patient_id_cols if col in df.columns), None)
    
    kpis = {}
    
    # Outcome score metrics
    if outcome_col:
        kpis["avg_outcome_score"] = safe_mean(df, outcome_col)
    
    # Readmission metrics
    if readmit_col:
        kpis["readmission_rate"] = safe_ratio(df[readmit_col].sum() if readmit_col in df.columns else 0, len(df))
    
    # Length of stay metrics
    if los_col:
        kpis["avg_length_of_stay"] = safe_mean(df, los_col)
        kpis["total_patient_days"] = safe_sum(df, los_col)
    
    # Patient count
    if patient_col:
        kpis["total_patients"] = len(df[patient_col].unique()) if patient_col in df.columns else len(df)
    
    # Fallback: return empty dict if no relevant columns found
    if not kpis:
        kpis = {
            "avg_outcome_score": None,
            "readmission_rate": None,
            "avg_length_of_stay": None,
            "total_patients": None
        }
    
    return kpis
