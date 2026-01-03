from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re


# =====================================================
# NORMALIZATION (CRITICAL)
# =====================================================

def _normalize(name: str) -> str:
    """
    Aggressively normalize column names to handle:
    - spelling mistakes
    - punctuation
    - spaces / dots / hyphens
    - casing
    """
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)   # remove punctuation
    name = re.sub(r"\s+", "_", name)      # spaces -> underscore
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# =====================================================
# UNIVERSAL SEMANTIC COLUMN MAP (EXPANDED)
# =====================================================

SEMANTIC_COLUMN_MAP: Dict[str, List[str]] = {

    # ---------- Identity ----------
    "id": ["id", "uid", "uuid"],
    "record_id": ["record_id", "row_id", "entry_id"],

    # ---------- Person / Entity ----------
    "patient_id": [
        "patient_id", "patientid", "pid", "ptid", "pt_id",
        "patient", "mrn", "uhid", "medical_record",
    ],
    "customer_id": [
        "customer_id", "customerid", "cust_id", "cid", "customer",
    ],
    "employee_id": [
        "employee_id", "emp_id", "staff_id", "eid",
    ],

    # ---------- Dates ----------
    "date": [
        "date", "created_date", "timestamp", "time", "event_date",
    ],
    "admission_date": [
        "admission_date", "admit_date", "admission",
    ],
    "discharge_date": [
        "discharge_date", "discharge",
    ],
    "order_date": [
        "order_date", "purchase_date",
    ],
    "fill_date": [
        "fill_date", "dispense_date", "rx_date",
    ],

    # ---------- Time / Duration ----------
    "duration": [
        "duration", "wait_time", "tat", "turnaround",
        "cycle_time", "delay",
    ],
    "length_of_stay": [
        "length_of_stay", "los", "stay_length", "lengthofstay",
    ],

    # ---------- Financial ----------
    "cost": [
        "cost", "charges", "charge", "billing",
        "bill_amount", "expense", "amount",
    ],
    "revenue": [
        "revenue", "sales", "turnover",
    ],
    "discount": [
        "discount", "discount_rate", "rebate",
    ],

    # ---------- Outcomes / Flags ----------
    "readmitted": [
        "readmitted", "readmit", "re_admitted", "no_show",
    ],
    "flag": [
        "flag", "indicator", "binary", "event", "outcome",
    ],
    "mortality": [
        "mortality", "death", "expired", "is_dead",
    ],

    # ---------- Categorical ----------
    "facility": [
        "facility", "hospital", "site", "location", "center",
    ],
    "doctor": [
        "doctor", "physician", "provider", "consultant", "clinician",
    ],
    "department": [
        "department", "dept", "unit", "team",
    ],
    "insurance": [
        "insurance", "payer", "insurance_provider",
    ],

    # ---------- Supply / Population ----------
    "supply": [
        "days_supply", "supply", "inventory_days",
    ],
    "population": [
        "population", "members", "covered_lives", "people",
    ],
}


# =====================================================
# COLUMN RESOLUTION ENGINE (v2)
# =====================================================

def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Optional[str]:
    """
    Backward-compatible resolver.
    Returns column name or None.
    """
    col, _ = resolve_column_with_confidence(df, semantic_key, cutoff)
    return col


def resolve_column_with_confidence(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Tuple[Optional[str], float]:
    """
    Authoritative resolver with confidence.

    Returns:
        (column_name | None, confidence 0–1)
    """

    if df is None or semantic_key not in SEMANTIC_COLUMN_MAP:
        return None, 0.0

    # Normalize dataframe columns
    norm_cols = {
        _normalize(c): c for c in df.columns
    }

    best_match = None
    best_score = 0.0

    for alias in SEMANTIC_COLUMN_MAP[semantic_key]:
        alias_norm = _normalize(alias)

        for norm_col, original_col in norm_cols.items():
            score = _similarity(alias_norm, norm_col)
            if score > best_score:
                best_score = score
                best_match = original_col

    if best_score >= cutoff:
        return best_match, round(best_score, 2)

    return None, 0.0
