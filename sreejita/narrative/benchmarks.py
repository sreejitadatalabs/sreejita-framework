# sreejita/narrative/benchmarks.py

"""
HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
--------------------------------------------------
Sources: CMS Inpatient Norms, OECD Health Statistics, HFMA.
Used to drive executive judgment logic.
"""

HEALTHCARE_BENCHMARKS = {
    # --- Operational Efficiency ---
    "avg_los": {
        "good": 5.0,        # Acute care standard
        "warning": 7.0,     # Efficiency drift
        "critical": 9.0,    # Capacity crisis
        "unit": "days",
        "source": "CMS Inpatient Norms"
    },
    
    # --- Clinical Quality ---
    "readmission_rate": {
        "good": 0.10,       # Top Decile
        "warning": 0.15,    # Median
        "critical": 0.20,   # Penalty Risk
        "unit": "rate",
        "source": "CMS Hospital Compare"
    },
    
    # --- Capacity ---
    "bed_turnover_index": {
        "target": 0.20,     # 1 patient every 5 days
        "source": "Ops Efficiency Standard"
    },
    
    # --- Financial (Multipliers of Median) ---
    "cost_per_patient": {
        "warning_multiplier": 1.5,
        "critical_multiplier": 2.5,
        "source": "Internal Financial Baseline"
    },

    # --- Workforce ---
    "provider_variance": {
        "good": 0.20,       # Standardized
        "warning": 0.40,    # Variable
        "critical": 0.60,   # Chaotic
        "source": "Clinical Ops"
    }
}

# Thresholds for Score Calculation
HEALTHCARE_THRESHOLDS = {
    "avg_los_warning": 6.0,
    "avg_los_critical": 7.0,
    "long_stay_rate_warning": 0.20,
    "long_stay_rate_critical": 0.30,
    "readmission_warning": 0.15,
    "readmission_critical": 0.18,
    "cost_multiplier_warning": 1.2,
    "cost_multiplier_critical": 1.5,
    "provider_variance_warning": 0.40,
    "weekend_rate_warning": 0.35,
}
