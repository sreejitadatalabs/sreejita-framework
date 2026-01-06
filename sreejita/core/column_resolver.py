# =====================================================
# COLUMN RESOLVER — UNIVERSAL SEMANTIC ENGINE (FINAL)
# Sreejita Framework v3.5.x
# =====================================================

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# =====================================================
# SEMANTIC COLUMN MAP (AUTHORITATIVE, NON-OVERLAPPING)
# =====================================================

SEMANTIC_COLUMN_MAP: Dict[str, Dict[str, Any]] = {

    # ---------------- IDENTITY ----------------
    "patient_id": {
        "aliases": [
            "patient_id", "patientid", "pid", "ptid",
            "mrn", "uhid", "medical_record_number"
        ],
        "dtype": "object",
    },

    "encounter": {
        "aliases": [
            "encounter_id", "visit_id", "case_id",
        "appointment_id", "appt_id", "claim_id",
        "test_id", "order_id", "transaction_id",
        "service_id", "record_id"
        ],
        "dtype": "object",
        "priority": 0.85,
    },

    # ---------------- TIME (STRICT) ----------------
    # ⚠ DO NOT add generic "time" here
    "admission_date": {
        "aliases": [
            "admission_date", "admit_date", "admission_datetime", "date_of_admission", "admitted_on", "admitted_date",
        # Generic (hospital context)
        "visit_date", "visitdate", "encounter_date",
        "check_in_date", "checkin_date", "start_date"
        ],
        "dtype": "datetime",
        "priority": 0.95,
    },

    "discharge_date": {
        "aliases": [
            "discharge_date", "discharged_at", "discharged_date",
        "discharge_datetime", "date_of_discharge", "checkout_date",
        "end_date", "completion_date"
        ],
        "dtype": "datetime",
        "priority": 0.92,
    },

    "fill_date": {
        "aliases": [
            "fill_date", "dispense_date", "rx_filled_date",
        "dispensed_date", "medication_fill_date", "refill_date",
        "prescription_fill_date", "filled_on", "dispensed_on",
        "pickup_date", "drug_fill_date"
        ],
        "dtype": "datetime",
        "priority": 0.90,
    },

    # ---------------- DURATIONS ----------------
    "length_of_stay": {
        "aliases": [
            "length_of_stay", "los", "stay_length"
        ],
        "dtype": "numeric",
    },

    "duration": {
        "aliases": [
            "duration", "duration_minutes", "duration_hours",
        # Clinical-specific
        "wait_time", "waiting_time", "patient_wait_time",
        "turnaround_time", "tat", "turnaround_minutes",
        # Clinic/pharmacy
        "appointment_duration", "visit_duration", "service_time",
        "checkout_time", "time_in_clinic", "chair_time"
        ],
        "dtype": "numeric",
        "priority": 0.85,
    },

    # ---------------- COST (NEUTRAL) ----------------
    "cost": {
        "aliases": [
            "cost", "billing_amount", "total_charges", "charges"
        ],
        "dtype": "numeric",
    },

    # ---------------- FLAGS ----------------
    "readmitted": {
        "aliases": [
            "readmitted", "readmission_flag"
        ],
        "dtype": "binary",
    },

    "flag": {
        "aliases": [
            "flag", "outcome", "event_flag",
        "mortality", "no_show", "readmission",
        "critical_result", "alert_flag", "result_flag",
        "specimen_rejection", "device_alert", "safety_event",
        "yes_no_flag", "binary_outcome"
        ],
        "dtype": "binary",
        "priority": 0.80,
    },

    "diagnosis": {
        "aliases": [
            "diagnosis", "diagnoses", "primary_diagnosis",
            "diagnosis_code", "icd_code", "icd9", "icd10",
            "disease_code", "condition_code", "condition"
        ],
        "dtype": "object",
        "priority": 0.88,
    },

    # ---------------- STRUCTURE ----------------
    "facility": {
        "aliases": [
            "facility", "hospital", "clinic", "location", "branch"
        ],
        "dtype": "object",
    },

    "doctor": {
        "aliases": [
            "doctor", "physician", "provider"
        ],
        "dtype": "object",
    },

    "bed_id": {
        "aliases": [
            "bed_id", "bed", "room_no"
        ],
        "dtype": "object",
    },

    # ---------------- PHARMACY / POPULATION ----------------
    "supply": {
        "aliases": [
            "days_supply", "supply", "days_on_hand",
        "quantity_dispensed", "quantity", "qty",
        "quantity_supplied", "refill_days", "pill_count",
        "dose_count", "num_units"
        ],
        "dtype": "numeric",
        "priority": 0.82,
    },

    "population": {
        "aliases": [
            "population", "covered_lives"
        ],
        "dtype": "numeric",
    },
}

# =====================================================
# NORMALIZATION HELPERS
# =====================================================

def _normalize(name: str) -> str:
    """Normalize column names for fuzzy matching"""
    name = str(name).lower().strip()
    # Remove special characters but preserve structure
    name = re.sub(r"[^\w\s]", "", name)
    # Collapse spaces to underscores
    name = re.sub(r"\s+", "_", name)
    # Remove common prefixes/suffixes
    name = re.sub(r"^(col_|column_)", "", name)
    name = re.sub(r"(_col|_column)$", "", name)
    # Remove redundant underscores
    name = re.sub(r"_+", "_", name)
    return name

def _similarity(a: str, b: str) -> float:
    """Enhanced similarity with substring matching"""
    ratio = SequenceMatcher(None, a, b).ratio()
    
    # Bonus for exact substring matches
    if a in b or b in a:
        ratio = max(ratio, 0.85)
    
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
# CORE RESOLVER (CONFIDENCE-AWARE)
# =====================================================

def resolve_column_with_confidence(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.70,
) -> Tuple[Optional[str], float]:

    if df is None or df.empty or semantic_key not in SEMANTIC_COLUMN_MAP:
        return None, 0.0

    spec = SEMANTIC_COLUMN_MAP[semantic_key]
    aliases = spec["aliases"]
    expected_dtype = spec.get("dtype")

    norm_cols = {_normalize(c): c for c in df.columns}

    best_col = None
    best_score = 0.0

    for alias in aliases:
        alias_norm = _normalize(alias)

        for norm_col, original_col in norm_cols.items():
            name_score = _similarity(alias_norm, norm_col)
            if name_score < 0.65:
                continue

            series = df[original_col]
            coverage = _coverage(series)
            if coverage < 0.30:
                continue

            dtype_score = _dtype_score(series, expected_dtype)

            # Get priority multiplier from semantic map (default 1.0)
            priority = spec.get("priority", 1.0)
            
            # Name score is domain-critical
            final_score = (
                name_score * 0.55 * priority +
                dtype_score * 0.30 +
                coverage * 0.15
            ) / priority  # Normalize back

            if final_score > best_score:
                best_score = final_score
                best_col = original_col

    if best_score >= cutoff:
        return best_col, round(best_score, 2)

    return None, 0.0


# =====================================================
# BACKWARD COMPATIBLE API
# =====================================================

def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.70,
) -> Optional[str]:
    col, _ = resolve_column_with_confidence(df, semantic_key, cutoff)
    return col


def bulk_resolve(
    df: pd.DataFrame,
    keys: List[str],
) -> Dict[str, Optional[str]]:
    return {k: resolve_column(df, k) for k in keys}


# =====================================================
# SEMANTIC CAPABILITY SIGNALS (DOMAIN-SAFE)
# =====================================================

def resolve_semantics(df: pd.DataFrame) -> Dict[str, bool]:
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
