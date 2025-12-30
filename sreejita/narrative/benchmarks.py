# sreejita/narrative/benchmarks.py

"""
HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
--------------------------------------------------
This file defines "What Good Looks Like" for the Healthcare domain.
It includes:
1. BENCHMARKS: Rich metadata for narrative context (Source, Unit, Targets).
2. THRESHOLDS: Flat key-value pairs for code logic/scoring.
3. EXTERNAL LIMITS: Governance anchors to prevent internal bias (The "Reality Check").
"""

# =====================================================
# 1. NARRATIVE BENCHMARKS (Context & Sources)
# =====================================================
HEALTHCARE_BENCHMARKS = {
    # --- Operational Efficiency ---
    "avg_los": {
        "good": 5.0,        # Target: 5.0 days (Acute care standard)
        "warning": 7.0,     # Alert level
        "critical": 9.0,    # Crisis level
        "unit": "days",
        "source": "CMS Inpatient Norms"
    },
    
    # --- Clinical Quality ---
    "readmission_rate": {
        "good": 0.10,       # Target: 10%
        "warning": 0.15,    # Alert level
        "critical": 0.20,   # Crisis level (>20% risks penalties)
        "unit": "rate",
        "source": "CMS Hospital Compare"
    },
    
    # --- Capacity Management ---
    "bed_turnover_index": {
        "good": 0.20,       # Target: ~1 patient every 5 days
        "warning": 0.14,    # Slow turnover
        "critical": 0.10,   # Bed blocking
        "unit": "index",
        "source": "Ops Efficiency Std"
    },
    
    # --- Clinical Variation ---
    "provider_variance_score": {
        "good": 0.20,       # Low variation (Standardized care)
        "warning": 0.35,    # Moderate variation
        "critical": 0.50,   # High variation (Quality risk)
        "unit": "cv",       # Coefficient of Variation
        "source": "Clinical Variation Std"
    },
    
    # --- Financial Health (Dynamic) ---
    "cost_per_patient": {
        "warning_multiplier": 1.2,   # Tightened from 1.5
        "critical_multiplier": 1.5,  # Tightened from 2.5
        "source": "Internal Financial Baseline"
    }
}


# =====================================================
# 2. LOGIC THRESHOLDS (For Scoring & Alerts)
# =====================================================
HEALTHCARE_THRESHOLDS = {
    # Operations
    "avg_los_warning": 6.0,
    "avg_los_critical": 7.0,
    "long_stay_rate_warning": 0.20,
    "long_stay_rate_critical": 0.30,

    # Clinical
    "readmission_warning": 0.15,
    "readmission_critical": 0.18,

    # Financial
    "cost_multiplier_warning": 1.2,   # vs median
    "cost_multiplier_critical": 1.5,

    # Workforce / Capacity
    "provider_variance_warning": 0.40,
    "weekend_rate_warning": 0.35,
}

# =====================================================
# 3. EXTERNAL GOVERNANCE LIMITS (The "Reality Check")
# =====================================================
# These prevent "Internal Inflation" (where high internal costs set high internal benchmarks).
# The system forces the benchmark down to these caps if the internal data is too high.
HEALTHCARE_EXTERNAL_LIMITS = {
    "avg_cost_per_patient": {
        "soft_cap": 12000,
        "hard_cap": 20000,
        "source": "CMS / OECD blended heuristic"
    },
    "avg_los": {
        "soft_cap": 5.0,
        "hard_cap": 10.0,
        "source": "Standard Acute Care Norms"
    }
}
