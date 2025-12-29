# sreejita/narrative/benchmarks.py

"""
HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
--------------------------------------------------
This file defines "What Good Looks Like" for the Healthcare domain.
It is used by the Narrative Engine to generate judgment (Critical/Warning/Good).

Sources:
- CMS (Centers for Medicare & Medicaid Services)
- OECD Health Statistics
- Standard Operational Heuristics for Acute Care
"""

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
    # These are multipliers of the dataset median, not absolute dollars.
    "cost_per_patient": {
        "warning_multiplier": 1.5,   # 1.5x Median = Warning
        "critical_multiplier": 2.5,  # 2.5x Median = Critical Outlier
        "source": "Internal Financial Baseline"
    }
}
