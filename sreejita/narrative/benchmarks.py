# sreejita/narrative/benchmarks.py

# =====================================================
# HEALTHCARE INTELLIGENCE STANDARDS
# Source: OECD / CMS / Internal Ops Heuristics
# =====================================================

HEALTHCARE_BENCHMARKS = {
    "avg_los": {
        "good": 5.0,
        "warning": 7.0,
        "critical": 9.0,
        "unit": "days",
        "source": "CMS Inpatient Norms"
    },
    "readmission_rate": {
        "good": 0.10,
        "warning": 0.15,
        "critical": 0.20,
        "unit": "rate",
        "source": "CMS Hospital Compare"
    },
    "bed_turnover_index": {
        "good": 0.20, # 1 patient every 5 days
        "warning": 0.14,
        "critical": 0.10,
        "unit": "index",
        "source": "Ops Efficiency Std"
    },
    "provider_variance_score": {
        "good": 0.20,
        "warning": 0.35,
        "critical": 0.50,
        "unit": "cv",
        "source": "Clinical Variation Std"
    },
    # Dynamic Financial Benchmarks (Multipliers of Median)
    "cost_per_patient": {
        "warning_multiplier": 1.5, 
        "critical_multiplier": 2.5,
        "source": "Internal Financial Baseline"
    }
}
