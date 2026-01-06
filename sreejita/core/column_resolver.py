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
            "encounter_id", "visit_id", "case_id"
        ],
        "dtype": "object",
    },

    # ---------------- TIME (STRICT) ----------------
    # ⚠ DO NOT add generic "time" here
    "admission_date": {
        "aliases": [
            "admission_date", "admit_date",
            "date_of_admission"
        ],
        "dtype": "datetime",
    },

    "discharge_date": {
        "aliases": [
            "discharge_date", "discharged_at"
        ],
        "dtype": "datetime",
    },

    "fill_date": {
        "aliases": [
            "fill_date", "dispense_date", "rx_filled_date"
        ],
        "dtype": "datetime",
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
            "duration", "wait_time", "turnaround_time", "tat"
        ],
        "dtype": "numeric",
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
            "flag", "outcome", "event_flag"
        ],
        "dtype": "binary",
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
            "days_supply", "supply"
        ],
        "dtype": "numeric",
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
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


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

            final_score = (
                name_score * 0.55 +
                dtype_score * 0.30 +
                coverage * 0.15
            )

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
