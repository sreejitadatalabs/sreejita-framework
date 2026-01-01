# sreejita/narrative/benchmarks.py

"""
HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
--------------------------------------------------
This file defines "What Good Looks Like" for the Healthcare domain.

It contains:
1. BENCHMARKS        → Narrative context (targets, sources, units)
2. THRESHOLDS        → Flat logic thresholds (alerts & scoring)
3. EXTERNAL LIMITS   → Governance caps to prevent internal bias
4. ACCESS HELPERS    → Canonical, safe access for the entire framework
"""

from typing import Dict, Any


# =====================================================
# 1. NARRATIVE BENCHMARKS (Context & Sources)
# =====================================================

HEALTHCARE_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # --- Operational Efficiency ---
    "avg_los": {
        "good": 5.0,
        "warning": 7.0,
        "critical": 9.0,
        "unit": "days",
        "source": "CMS Inpatient Norms",
    },

    # --- Clinical Quality ---
    "readmission_rate": {
        "good": 0.10,
        "warning": 0.15,
        "critical": 0.20,
        "unit": "rate",
        "source": "CMS Hospital Compare",
    },

    # --- Capacity Management ---
    "bed_turnover_index": {
        "good": 0.20,
        "warning": 0.14,
        "critical": 0.10,
        "unit": "index",
        "source": "Operational Efficiency Standard",
    },

    # --- Clinical Variation ---
    "provider_variance_score": {
        "good": 0.20,
        "warning": 0.35,
        "critical": 0.50,
        "unit": "cv",
        "source": "Clinical Variation Standard",
    },

    # --- Financial Health ---
    "cost_per_patient": {
        "warning_multiplier": 1.2,
        "critical_multiplier": 1.5,
        "unit": "currency",
        "source": "Internal Financial Baseline",
    },
}


# =====================================================
# 2. LOGIC THRESHOLDS (Scoring & Alerts)
# =====================================================

HEALTHCARE_THRESHOLDS: Dict[str, float] = {
    # Operations
    "avg_los_warning": 6.0,
    "avg_los_critical": 7.0,
    "long_stay_rate_warning": 0.20,
    "long_stay_rate_critical": 0.30,

    # Clinical
    "readmission_warning": 0.15,
    "readmission_critical": 0.18,

    # Financial
    "cost_multiplier_warning": 1.2,
    "cost_multiplier_critical": 1.5,

    # Workforce / Capacity
    "provider_variance_warning": 0.40,
    "weekend_rate_warning": 0.35,
}


# =====================================================
# 3. EXTERNAL GOVERNANCE LIMITS (Reality Anchors)
# =====================================================

HEALTHCARE_EXTERNAL_LIMITS: Dict[str, Dict[str, Any]] = {
    "avg_cost_per_patient": {
        "soft_cap": 12000,
        "hard_cap": 20000,
        "source": "CMS / OECD blended heuristic",
    },
    "avg_los": {
        "soft_cap": 5.0,
        "hard_cap": 10.0,
        "source": "Standard Acute Care Norms",
    },
}


# =====================================================
# 4. ACCESS HELPERS (CANONICAL API)
# =====================================================

def get_benchmark(metric: str) -> Dict[str, Any]:
    """
    Returns narrative benchmark metadata for a metric.
    """
    return HEALTHCARE_BENCHMARKS.get(metric, {})


def get_threshold(key: str, default: float = None) -> float:
    """
    Safe accessor for flat logic thresholds.
    """
    return HEALTHCARE_THRESHOLDS.get(key, default)


def apply_external_limits(metric: str, value: float) -> float:
    """
    Enforces governance caps on KPI values to prevent internal inflation.
    """
    if value is None:
        return value

    limits = HEALTHCARE_EXTERNAL_LIMITS.get(metric)
    if not limits:
        return value

    soft = limits.get("soft_cap")
    hard = limits.get("hard_cap")

    if hard is not None:
        value = min(value, hard)
    elif soft is not None:
        value = min(value, soft)

    return value


def explain_external_limit(metric: str) -> str:
    """
    Returns governance source for an external limit, if any.
    """
    limit = HEALTHCARE_EXTERNAL_LIMITS.get(metric, {})
    return limit.get("source", "")


# =====================================================
# END OF FILE — AUTHORITATIVE TRUTH LAYER
# =====================================================
