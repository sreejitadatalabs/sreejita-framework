# =====================================================
# COLUMN RESOLVER — UNIVERSAL SEMANTIC ENGINE (LOCKED)
# Sreejita Framework v3.6 STABILIZED
# =====================================================

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# =====================================================
# SEMANTIC COLUMN MAP (STRICT, NON-POLLUTING)
# =====================================================
# RULES:
# - No alias may imply a domain by itself
# - Pharmacy signals must be EXPLICIT
# - No generic time/cost leakage
# =====================================================

SEMANTIC_COLUMN_MAP: Dict[str, Dict[str, Any]] = {

    # ---------------- IDENTITY ----------------
    "patient_id": {
        "aliases": [
            "patient_id", "patientid", "pid", "ptid",
            "patient", "patient_code",
            "mrn", "uhid", "medical_record_number"
        ],
        "dtype": "object",
        "priority": 1.0,
    },

    "encounter": {
        "aliases": [
            "encounter_id", "visit_id", "case_id",
            "appointment_id", "appt_id", "test_id"
        ],
        "dtype": "object",
        "priority": 0.9,
    },

    # ---------------- TIME (STRICT) ----------------
    "admission_date": {
        "aliases": [
            "admission_date", "admit_date",
            "admission_datetime", "date_of_admission",
            "admissiondate", "admit_dt", "visitdate",
            "visit_date", "encounter_date", "check_in_date"
            "admission date", "admit date",
        ],
        "dtype": "datetime",
        "priority": 1.0,
    },

    "discharge_date": {
        "aliases": [
            "discharge_date", "discharged_date",
            "dischargedate", "disch_date", "disch_dt",
            "discharge_datetime", "date_of_discharge",
            "discharge date", "discharged date",
            "checkout_date", "completion_date", "end_date"
        ],
        "dtype": "datetime",
        "priority": 0.95,
    },

    # 🚨 PHARMACY TIME — EXPLICIT ONLY
    "fill_date": {
        "aliases": [
            "fill_date", "dispense_date",
            "rx_filled_date", "refill_date",
            "prescription_fill_date"
        ],
        "dtype": "datetime",
        "priority": 0.95,
    },

    # ---------------- DURATION ----------------
    "length_of_stay": {
        "aliases": ["length_of_stay", "los", "stay_days", "days_stay", "stay_length", "length of stay", "stay length"],
        "dtype": "numeric",
        "priority": 1.0,
    },

    "duration": {
        "aliases": [
            "duration", "duration_minutes",
            "wait_time", "turnaround_time",
            "appointment_duration", "visit_duration"
        ],
        "dtype": "numeric",
        "priority": 0.9,
    },

    # ---------------- COST ----------------
    "cost": {
        "aliases": [
            "cost", "billing_amount",
            "total_charges", "charges", "billing amount", "total charges"
        ],
        "dtype": "numeric",
        "priority": 0.9,
    },

    # ---------------- FLAGS (GENERIC ONLY) ----------------
    # IMPORTANT:
    # - This does NOT imply mortality, alerts, or pharmacy
    # - Interpretation is domain responsibility
    "flag": {
        "aliases": [
            "flag", "outcome", "event_flag",
            "binary_outcome", "yes_no_flag"
        ],
        "dtype": "binary",
        "priority": 0.85,
    },

    "readmitted": {
        "aliases": [
            "readmitted", "readmission_flag"
        ],
        "dtype": "binary",
        "priority": 0.95,
    },

    # ---------------- STRUCTURE ----------------
    "facility": {
        "aliases": [
            "facility", "hospital", "clinic",
                    "location", "dept", "department", "branch", "hospital branch"],
        "dtype": "object",
        "priority": 0.8,
    },
    "doctor": {
        "aliases": [
            "doctor", "physician", "provider"
        ],
        "dtype": "object",
        "priority": 0.8,
    },
    "diagnosis": 
    {
        "aliases": [
            "diagnosis", "icd_code", "primary_diagnosis", "diagnosis_code"
        ],
        "dtype": "object",
        "priority": 0.9,
    },

    "bed_id": {
        "aliases": [
            "bed_id", "bed", "room_no"
        ],
        "dtype": "object",
        "priority": 0.9,
    },

    "admission_type": {
        "aliases": [
            "admission_type", "admissiontype", "admit_type", "admittype",
            "admission_class", "visit_type", "encounter_type"
        ],
        "dtype": "object",
        "priority": 0.9,
    },

    # ---------------- PHARMACY / POPULATION (HARD-GATED) ----------------
    # NOTE:
    # These columns NEVER imply pharmacy alone.
    # Sub-domain logic must require MULTIPLE signals.
    "supply": {
        "aliases": [
            "days_supply", "supply",
            "days_on_hand", "quantity_dispensed"
        ],
        "dtype": "numeric",
        "priority": 0.9,
    },

    "population": {
        "aliases": [
            "population", "covered_lives"
        ],
        "dtype": "numeric",
        "priority": 0.9,
    },
}

# =====================================================
# NORMALIZATION HELPERS (LOCKED)
# =====================================================

def _normalize(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name


def _similarity(a: str, b: str) -> float:
    ratio = SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        ratio = max(ratio, 0.90)
    return ratio


def _coverage(series: pd.Series) -> float:
    return float(series.notna().mean())


def _dtype_score(series: pd.Series, expected: str) -> float:
    try:
        if expected == "numeric":
            return 1.0 if pd.to_numeric(series, errors="coerce").notna().mean() > 0.7 else 0.0

        if expected == "datetime":
            return 1.0 if pd.to_datetime(series, errors="coerce").notna().mean() > 0.7 else 0.0

        if expected == "binary":
            uniq = pd.to_numeric(series, errors="coerce").dropna().unique()
            return 1.0 if set(uniq).issubset({0, 1}) else 0.0

        return 1.0
    except Exception:
        return 0.0

# =====================================================
# CORE RESOLVER (DETERMINISTIC, NON-HALLUCINATING)
# =====================================================

def resolve_column_with_confidence(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Tuple[Optional[str], float]:

    if df is None or df.empty:
        return None, 0.0

    if semantic_key not in SEMANTIC_COLUMN_MAP:
        return None, 0.0

    spec = SEMANTIC_COLUMN_MAP[semantic_key]
    aliases = spec["aliases"]
    expected_dtype = spec.get("dtype")
    priority = spec.get("priority", 1.0)

    norm_cols = {_normalize(c): c for c in df.columns}

    best_col: Optional[str] = None
    best_score: float = 0.0

    for alias in aliases:
        alias_norm = _normalize(alias)

        for norm_col, original_col in norm_cols.items():
            name_score = _similarity(alias_norm, norm_col)
            if name_score < 0.70:
                continue

            series = df[original_col]
            coverage = _coverage(series)
            if coverage < 0.30:
                continue

            dtype_score = _dtype_score(series, expected_dtype)

            final_score = (
                name_score * 0.55 +
                dtype_score * 0.30 +
                coverage * 0.15
            ) * priority

            if final_score > best_score:
                best_score = final_score
                best_col = original_col

    if best_score >= cutoff:
        return best_col, round(best_score, 2)

    return None, 0.0

# =====================================================
# BACKWARD-COMPATIBLE API
# =====================================================

def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.72,
) -> Optional[str]:
    col, _ = resolve_column_with_confidence(df, semantic_key, cutoff)
    return col


def bulk_resolve(
    df: pd.DataFrame,
    keys: List[str],
) -> Dict[str, Optional[str]]:
    return {k: resolve_column(df, k) for k in keys}

# =====================================================
# SEMANTIC CAPABILITY SIGNALS (SAFE, NON-DOMAINAL)
# =====================================================

def resolve_semantics(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Capability signals ONLY.
    Never infer domain or sub-domain here.
    """
    return {
        "has_patient_id": bool(resolve_column(df, "patient_id")),
        "has_admission_date": bool(resolve_column(df, "admission_date")),
        "has_discharge_date": bool(resolve_column(df, "discharge_date")),
        "has_los": bool(resolve_column(df, "length_of_stay")),
        "has_duration": bool(resolve_column(df, "duration")),
        "has_cost": bool(resolve_column(df, "cost")),
        "has_supply": bool(resolve_column(df, "supply")),
        "has_population": bool(resolve_column(df, "population")),
    }
