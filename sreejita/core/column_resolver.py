import re
import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

# =====================================================
# UNIVERSAL SEMANTIC COLUMN MAP (AUTHORITATIVE)
# =====================================================
# Expanding this map increases detection strength
# across ALL domains without touching domain logic.

SEMANTIC_COLUMN_MAP: Dict[str, List[str]] = {
    # ---------- Identity ----------
    "id": ["id", "uid", "uuid", "key", "pk"],
    "record_id": ["record_id", "row_id", "entry_id", "transaction_id"],

    # ---------- Person / Entity ----------
    "patient_id": [
        "patient_id", "patientid", "pid", "ptid", "pt_id",
        "patient", "mrn", "uhid", "medical_record_number", "member_id"
    ],
    "customer_id": [
        "customer_id", "customerid", "cust_id", "cid", "customer", "client_id"
    ],
    "employee_id": [
        "employee_id", "emp_id", "staff_id", "eid", "provider_id"
    ],

    # ---------- Dates ----------
    "date": [
        "date", "created_date", "timestamp", "time",
        "event_date", "datetime"
    ],
    "admission_date": [
        "admission_date", "admit_date", "admission",
        "admitted_at", "date_of_admission"
    ],
    "discharge_date": [
        "discharge_date", "discharge", "discharged_at"
    ],
    "order_date": [
        "order_date", "purchase_date", "transaction_date"
    ],
    "fill_date": [
        "fill_date", "dispense_date", "rx_date", "prescribed_date"
    ],

    # ---------- Time / Duration ----------
    "duration": [
        "duration", "wait_time", "tat", "turnaround",
        "cycle_time", "delay", "processing_time"
    ],
    "length_of_stay": [
        "length_of_stay", "los", "stay_length",
        "lengthofstay", "days_in_hospital"
    ],

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
    "facility": [
        "facility", "hospital", "site", "location",
        "center", "clinic_name", "ward"
    ],
    "doctor": [
        "doctor", "physician", "provider",
        "consultant", "clinician", "md"
    ],
    "department": [
        "department", "dept", "unit", "team", "specialty"
    ],
    "insurance": [
        "insurance", "payer", "insurance_provider", "plan_name"
    ],

    # ---------- Supply / Population ----------
    "supply": [
        "days_supply", "supply", "inventory_days",
        "qty_on_hand", "stock_level"
    ],
    "population": [
        "population", "members", "covered_lives",
        "people", "census"
    ],

    # ---------- Hospital-Specific Structural Signals ----------
    "bed_id": [
        "room_number", "bed_id", "unit_id",
        "ward", "room_no", "bed_no"
    ],
    "admission_type": [
        "admission_type", "admit_type", "entry_mode", "urgency"
    ],
}

# =====================================================
# INTERNAL HELPERS
# =====================================================

def _normalize(name: str) -> str:
    """
    Normalize column names for robust matching.
    Example: 'Pat. ID #' -> 'pat_id'
    """
    if not isinstance(name, str):
        return str(name)
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name

def _similarity(a: str, b: str) -> float:
    """Fuzzy similarity score (0–1)."""
    return SequenceMatcher(None, a, b).ratio()

# =====================================================
# COLUMN RESOLUTION ENGINE
# =====================================================

def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Optional[str]:
    """
    Resolve semantic intent → actual column name.
    """
    col, _ = resolve_column_with_confidence(df, semantic_key, cutoff)
    return col

def resolve_column_with_confidence(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Tuple[Optional[str], float]:
    """
    Resolve column with confidence score.

    Returns:
        (column_name | None, confidence 0.0–1.0)
    """
    if df is None or df.empty or semantic_key not in SEMANTIC_COLUMN_MAP:
        return None, 0.0

    norm_cols = { _normalize(c): c for c in df.columns }
    aliases = SEMANTIC_COLUMN_MAP[semantic_key]

    best_match: Optional[str] = None
    best_score: float = 0.0

    for alias in aliases:
        alias_norm = _normalize(alias)

        for norm_col, original_col in norm_cols.items():
            # Exact normalized match = immediate win
            if alias_norm == norm_col:
                return original_col, 1.0

            score = _similarity(alias_norm, norm_col)
            if score > best_score:
                best_score = score
                best_match = original_col

    if best_score >= cutoff:
        return best_match, round(best_score, 2)

    return None, 0.0

def bulk_resolve(
    df: pd.DataFrame,
    keys: List[str],
) -> Dict[str, Optional[str]]:
    """
    Resolve multiple semantic intents in one pass.
    """
    return {key: resolve_column(df, key) for key in keys}

# =====================================================
# SEMANTIC INFERENCE (STRUCTURAL SIGNALS)
# =====================================================
# ⚠️ NO KPI COMPUTATION HERE
# ⚠️ NO DOMAIN LOGIC HERE
# Only exposes dataset capabilities

def has_date_range(df: pd.DataFrame) -> bool:
    """
    Detect admission → discharge lifecycle.
    """
    admit = resolve_column(df, "admission_date")
    discharge = resolve_column(df, "discharge_date")
    return bool(admit and discharge)

def can_derive_los(df: pd.DataFrame) -> bool:
    """
    Determines whether LOS can be derived safely.
    """
    admit = resolve_column(df, "admission_date")
    discharge = resolve_column(df, "discharge_date")

    if not (admit and discharge):
        return False

    try:
        return (
            pd.to_datetime(df[admit], errors="coerce").notna().any()
            and pd.to_datetime(df[discharge], errors="coerce").notna().any()
        )
    except Exception:
        return False

def resolve_semantics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    High-level dataset capability signals.
    Used by domain engines for sub-domain detection.
    """
    return {
        "has_patient_id": bool(resolve_column(df, "patient_id")),
        "has_facility": bool(resolve_column(df, "facility")),
        "has_admission_discharge": has_date_range(df),
        "can_derive_los": can_derive_los(df),
        "has_bed_id": bool(resolve_column(df, "bed_id")),
        "has_admission_type": bool(resolve_column(df, "admission_type")),
    }
