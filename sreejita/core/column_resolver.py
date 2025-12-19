from difflib import get_close_matches

HEALTHCARE_SEMANTIC_MAP = {
    "patient_id": ["patient_id", "patientid", "pid", "patient"],
    "length_of_stay": ["length_of_stay", "los", "stay_length", "lengthofstay"],
    "readmitted": ["readmitted", "readmit", "re_admitted"],
    "mortality": ["mortality", "death", "expired", "is_dead"],
    "billing_amount": ["billing_amount", "bill_amount", "charges", "claim_amount", "cost"],
    "doctor": ["doctor", "physician", "provider"],
    "diagnosis": ["diagnosis", "dx", "condition"],
    "department": ["department", "dept", "unit"],
    "insurance": ["insurance", "payer", "insurance_provider"],
    "admission_date": ["admission_date", "admit_date", "admission"],
}

def resolve_column(df, semantic_key):
    candidates = HEALTHCARE_SEMANTIC_MAP.get(semantic_key, [])
    cols = list(df.columns)

    # Exact / lowercase match
    for c in cols:
        if c.lower() in candidates:
            return c

    # Fuzzy fallback
    matches = get_close_matches(
        semantic_key, cols, n=1, cutoff=0.75
    )
    return matches[0] if matches else None
