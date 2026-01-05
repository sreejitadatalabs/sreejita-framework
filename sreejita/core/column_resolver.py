# =====================================================
# COLUMN RESOLVER — UNIVERSAL SEMANTIC ENGINE (FINAL)
# Sreejita Framework v3.5.x
# =====================================================

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


# =====================================================
# SEMANTIC COLUMN MAP (AUTHORITATIVE)
# =====================================================

SEMANTIC_COLUMN_MAP: Dict[str, Dict[str, Any]] = {

    # ---------- Identity ----------
    "patient_id": {
        "aliases": [
            "patient_id", "patientid", "pid", "ptid", "pt_id",
            "patient", "mrn", "uhid", "medical_record_number", "member_id"
        ],
        "dtype": "object",
    },

    # ---------- Dates ----------
    "date": {
        "aliases": [
            "date", "created_date", "event_date",
            "timestamp", "datetime", "time"
        ],
        "dtype": "datetime",
    },

    "admission_date": {
        "aliases": [
            "admission_date", "admit_date",
            "admitted_at", "date_of_admission"
        ],
        "dtype": "datetime",
    },

    "discharge_date": {
        "aliases": [
            "discharge_date", "discharged_at", "discharge"
        ],
        "dtype": "datetime",
    },

    "fill_date": {
        "aliases": [
            "fill_date", "dispense_date", "rx_date", "prescribed_date"
        ],
        "dtype": "datetime",
    },

    # ---------- Time / Flow ----------
    "length_of_stay": {
        "aliases": [
            "length_of_stay", "los", "stay_length",
            "lengthofstay", "days_in_hospital"
        ],
        "dtype": "numeric",
    },

    "duration": {
        "aliases": [
            "duration", "wait_time", "tat",
            "turnaround", "cycle_time", "delay"
        ],
        "dtype": "numeric",
    },

    # ---------- Financial ----------
    "cost": {
        "aliases": [
            "cost", "charges", "charge", "billing",
            "bill_amount", "expense", "amount", "total_cost"
        ],
        "dtype": "numeric",
    },

    # ---------- Outcomes / Flags ----------
    "readmitted": {
        "aliases": [
            "readmitted", "readmit", "re_admitted", "is_readmission"
        ],
        "dtype": "binary",
    },

    "flag": {
        "aliases": [
            "flag", "indicator", "event", "outcome", "status_flag"
        ],
        "dtype": "binary",
    },

    # ---------- Structural ----------
    "facility": {
        "aliases": [
            "facility", "hospital", "site",
            "location", "center", "clinic", "ward"
        ],
        "dtype": "object",
    },

    "doctor": {
        "aliases": [
            "doctor", "physician", "provider",
            "consultant", "clinician"
        ],
        "dtype": "object",
    },

    "bed_id": {
        "aliases": [
            "bed_id", "bed_no", "room_no",
            "room_number", "unit_id"
        ],
        "dtype": "object",
    },

    "supply": {
        "aliases": [
            "days_supply", "supply",
            "inventory_days", "qty_on_hand"
        ],
        "dtype": "numeric",
    },

    "population": {
        "aliases": [
            "population", "members",
            "covered_lives", "census"
        ],
        "dtype": "numeric",
    },
}


# =====================================================
# NORMALIZATION HELPERS
# =====================================================

def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _dtype_score(series: pd.Series, expected: str) -> float:
    """
    Penalize incorrect dtypes.
    """
    try:
        if expected == "numeric":
            return 1.0 if pd.to_numeric(series, errors="coerce").notna().mean() > 0.6 else 0.5
        if expected == "datetime":
            return 1.0 if pd.to_datetime(series, errors="coerce").notna().mean() > 0.6 else 0.5
        if expected == "binary":
            uniq = series.dropna().unique()
            return 1.0 if len(uniq) <= 3 else 0.6
        return 1.0
    except Exception:
        return 0.5


def _coverage_score(series: pd.Series) -> float:
    """
    Penalize sparse columns.
    """
    non_null = series.notna().mean()
    if non_null >= 0.8:
        return 1.0
    if non_null >= 0.5:
        return 0.7
    return 0.4


# =====================================================
# CORE RESOLVER
# =====================================================

def resolve_column_with_confidence(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.65,
) -> Tuple[Optional[str], float]:

    if (
        df is None
        or df.empty
        or semantic_key not in SEMANTIC_COLUMN_MAP
    ):
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

            if name_score < 0.5:
                continue

            series = df[original_col]

            dtype_score = _dtype_score(series, expected_dtype)
            coverage_score = _coverage_score(series)

            final_score = (
                name_score * 0.6 +
                dtype_score * 0.25 +
                coverage_score * 0.15
            )

            if final_score > best_score:
                best_score = final_score
                best_col = original_col

    if best_score >= cutoff:
        return best_col, round(best_score, 2)

    return None, 0.0


def resolve_column(
    df: pd.DataFrame,
    semantic_key: str,
    cutoff: float = 0.65,
) -> Optional[str]:
    col, _ = resolve_column_with_confidence(df, semantic_key, cutoff)
    return col


def bulk_resolve(
    df: pd.DataFrame,
    keys: List[str],
) -> Dict[str, Optional[str]]:
    return {k: resolve_column(df, k) for k in keys}


# =====================================================
# SEMANTIC CAPABILITY SIGNALS (NO KPIs)
# =====================================================

def has_date_range(df: pd.DataFrame) -> bool:
    return bool(
        resolve_column(df, "admission_date")
        and resolve_column(df, "discharge_date")
    )


def can_derive_los(df: pd.DataFrame) -> bool:
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
    Dataset capability signals (used for sub-domain detection).
    """
    return {
        "has_patient_id": bool(resolve_column(df, "patient_id")),
        "has_facility": bool(resolve_column(df, "facility")),
        "has_admission_discharge": has_date_range(df),
        "can_derive_los": can_derive_los(df),
        "has_bed_id": bool(resolve_column(df, "bed_id")),
        "has_duration": bool(resolve_column(df, "duration")),
        "has_cost": bool(resolve_column(df, "cost")),
    }
