from difflib import get_close_matches
from typing import Dict, List, Optional
import pandas as pd


# =====================================================
# UNIVERSAL SEMANTIC COLUMN MAP
# =====================================================
# This map is DOMAIN-NEUTRAL by design.
# Domains may reuse keys but should NOT hardcode logic here.

SEMANTIC_COLUMN_MAP: Dict[str, List[str]] = {
    # ---------- Identity ----------
    "id": ["id", "uid", "uuid"],
    "record_id": ["record_id", "row_id", "entry_id"],

    # ---------- Person / Entity ----------
    "customer_id": ["customer_id", "customerid", "cust_id", "cid", "customer"],
    "patient_id": ["patient_id", "patientid", "pid", "patient"],
    "employee_id": ["employee_id", "emp_id", "staff_id", "eid"],

    # ---------- Dates ----------
    "date": ["date", "created_date", "timestamp"],
    "admission_date": ["admission_date", "admit_date", "admission"],
    "discharge_date": ["discharge_date", "discharge"],
    "order_date": ["order_date", "purchase_date"],

    # ---------- Quantities ----------
    "quantity": ["quantity", "qty", "units"],
    "length_of_stay": ["length_of_stay", "los", "stay_length", "lengthofstay"],

    # ---------- Financial ----------
    "billing_amount": ["billing_amount", "bill_amount", "charges", "claim_amount", "cost"],
    "revenue": ["revenue", "sales", "turnover"],
    "profit": ["profit", "margin", "net_profit"],
    "discount": ["discount", "discount_rate", "rebate"],

    # ---------- Outcomes / Status ----------
    "readmitted": ["readmitted", "readmit", "re_admitted"],
    "mortality": ["mortality", "death", "expired", "is_dead"],
    "status": ["status", "state", "result", "outcome"],

    # ---------- Categorical ----------
    "diagnosis": ["diagnosis", "dx", "condition"],
    "department": ["department", "dept", "unit", "team"],
    "doctor": ["doctor", "physician", "provider", "consultant"],
    "insurance": ["insurance", "payer", "insurance_provider"],
}


# =====================================================
# COLUMN RESOLUTION ENGINE
# =====================================================

def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.75
) -> Optional[str]:
    """
    Resolve a semantic column name to an actual dataframe column.

    Resolution strategy:
    1. Exact match (case-insensitive)
    2. Synonym match from SEMANTIC_COLUMN_MAP
    3. Fuzzy match fallback

    Args:
        df: Input dataframe
        semantic_key: Logical column intent (e.g., 'revenue', 'patient_id')
        cutoff: Similarity threshold for fuzzy matching

    Returns:
        Actual column name if resolved, else None

    Guarantees:
        - Deterministic
        - Domain-agnostic
        - Safe fallback behavior
    """

    if df is None or semantic_key is None:
        return None

    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    semantic_key = semantic_key.lower()

    # -----------------------------
    # 1. Exact match
    # -----------------------------
    if semantic_key in cols_lower:
        return cols_lower[semantic_key]

    # -----------------------------
    # 2. Synonym match
    # -----------------------------
    candidates = SEMANTIC_COLUMN_MAP.get(semantic_key, [])
    for c in cols:
        if c.lower() in candidates:
            return c

    # -----------------------------
    # 3. Fuzzy fallback (last resort)
    # -----------------------------
    matches = get_close_matches(
        semantic_key,
        cols_lower.keys(),
        n=1,
        cutoff=cutoff
    )

    return cols_lower[matches[0]] if matches else None
