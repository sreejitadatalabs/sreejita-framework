import re
import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# =====================================================
# UNIVERSAL SEMANTIC COLUMN MAP (HARDENED)
# =====================================================
# This map serves as the single source of truth for industry-standard aliases.
# Expanding these lists directly increases the detection strength of sub-domains.

SEMANTIC_COLUMN_MAP: Dict[str, List[str]] = {
    # ---------- Identity ----------
    "id": ["id", "uid", "uuid", "key", "pk"],
    "record_id": ["record_id", "row_id", "entry_id", "transaction_id"],

    # ---------- Person / Entity ----------
    "patient_id": [
        "patient_id", "patientid", "pid", "ptid", "pt_id",
        "patient", "mrn", "uhid", "medical_record_number", "member_id"
    ],
    "customer_id": ["customer_id", "customerid", "cust_id", "cid", "customer", "client_id"],
    "employee_id": ["employee_id", "emp_id", "staff_id", "eid", "provider_id"],

    # ---------- Dates ----------
    "date": ["date", "created_date", "timestamp", "time", "event_date", "datetime"],
    "admission_date": ["admission_date", "admit_date", "admission", "admitted_at", "date_of_admission"],
    "discharge_date": ["discharge_date", "discharge", "discharged_at"],
    "order_date": ["order_date", "purchase_date", "transaction_date"],
    "fill_date": ["fill_date", "dispense_date", "rx_date", "prescribed_date"],

    # ---------- Time / Duration ----------
    "duration": [
        "duration", "wait_time", "tat", "turnaround", 
        "cycle_time", "delay", "processing_time"
    ],
    "length_of_stay": ["length_of_stay", "los", "stay_length", "lengthofstay", "days_in_hospital"],

    # ---------- Financial ----------
    "cost": [
        "cost", "charges", "charge", "billing", 
        "bill_amount", "expense", "amount", "total_cost"
    ],
    "revenue": ["revenue", "sales", "turnover", "gross_amount"],
    "discount": ["discount", "discount_rate", "rebate", "markdown"],

    # ---------- Outcomes / Flags ----------
    "readmitted": ["readmitted", "readmit", "re_admitted", "is_readmission"],
    "flag": ["flag", "indicator", "binary", "event", "outcome", "status_flag"],
    "mortality": ["mortality", "death", "expired", "is_dead", "deceased"],
    "no_show": ["no_show", "missed_appointment", "is_noshow"],

    # ---------- Categorical ----------
    "facility": ["facility", "hospital", "site", "location", "center", "clinic_name", "ward"],
    "doctor": ["doctor", "physician", "provider", "consultant", "clinician", "md"],
    "department": ["department", "dept", "unit", "team", "specialty"],
    "insurance": ["insurance", "payer", "insurance_provider", "plan_name"],

    # ---------- Supply / Population ----------
    "supply": ["days_supply", "supply", "inventory_days", "qty_on_hand", "stock_level"],
    "population": ["population", "members", "covered_lives", "people", "census"],
}

# =====================================================
# INTERNAL HELPERS
# =====================================================

def _normalize(name: str) -> str:
    """
    Standardizes strings by removing punctuation, spaces, and casing.
    Example: 'Pat. ID #' -> 'pat_id'
    """
    if not isinstance(name, str):
        return str(name)
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)   # Remove punctuation
    name = re.sub(r"\s+", "_", name)      # Replace spaces with underscore
    return name

def _similarity(a: str, b: str) -> float:
    """Computes the ratio of similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

# =====================================================
# PUBLIC RESOLUTION ENGINE
# =====================================================

def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Optional[str]:
    """
    Standard resolver for mapping logic.
    Returns the original column name if found, else None.
    """
    col, _ = resolve_column_with_confidence(df, semantic_key, cutoff)
    return col

def resolve_column_with_confidence(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Tuple[Optional[str], float]:
    """
    Authoritative Resolver.
    Checks for exact normalized matches, synonyms, and then fuzzy similarity.
    
    Returns:
        (original_column_name | None, confidence_score 0.0 - 1.0)
    """
    if df is None or df.empty or semantic_key not in SEMANTIC_COLUMN_MAP:
        return None, 0.0

    # Cache normalized dataframe columns for mapping
    norm_cols = { _normalize(c): c for c in df.columns }
    
    target_aliases = SEMANTIC_COLUMN_MAP[semantic_key]
    
    best_match = None
    best_score = 0.0

    # Primary pass through semantic synonyms
    for alias in target_aliases:
        alias_norm = _normalize(alias)
        
        for norm_col, original_col in norm_cols.items():
            # Exact match (normalized) is immediate win
            if alias_norm == norm_col:
                return original_col, 1.0
            
            # Fuzzy match scoring
            score = _similarity(alias_norm, norm_col)
            if score > best_score:
                best_score = score
                best_match = original_col

    # Final threshold validation
    if best_score >= cutoff:
        return best_match, round(best_score, 2)

    return None, 0.0

def bulk_resolve(df: pd.DataFrame, keys: List[str]) -> Dict[str, Optional[str]]:
    """Resolves multiple semantic intents in a single pass."""
    return {key: resolve_column(df, key) for key in keys}
