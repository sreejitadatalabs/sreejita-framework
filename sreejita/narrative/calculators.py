# sreejita/narrative/calculators.py

from typing import Dict, Any


# =====================================================
# OPERATIONAL CALCULATIONS
# =====================================================

def calculate_excess_los_impact(kpis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates excess Length of Stay (LOS) impact.

    Expected KPI keys (best effort):
    - avg_los
    - target_los
    - annual_admissions
    """

    avg_los = kpis.get("avg_los")
    target_los = kpis.get("target_los")
    annual_admissions = kpis.get("annual_admissions")

    if not all(isinstance(x, (int, float)) for x in [avg_los, target_los, annual_admissions]):
        return {}

    excess_days_per_patient = max(avg_los - target_los, 0)
    excess_patient_days = excess_days_per_patient * annual_admissions

    return {
        "excess_days_per_patient": round(excess_days_per_patient, 2),
        "excess_patient_days": int(excess_patient_days),
    }


def calculate_readmission_impact(kpis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates readmission impact.

    Expected KPI keys:
    - readmission_rate
    - target_readmission_rate
    - annual_admissions
    """

    rate = kpis.get("readmission_rate")
    target = kpis.get("target_readmission_rate")
    admissions = kpis.get("annual_admissions")

    if not all(isinstance(x, (int, float)) for x in [rate, target, admissions]):
        return {}

    excess_rate = max(rate - target, 0)
    excess_cases = excess_rate * admissions

    return {
        "excess_readmission_rate": round(excess_rate * 100, 1),
        "excess_readmissions": int(excess_cases),
    }


# =====================================================
# FINANCIAL CALCULATIONS
# =====================================================

def estimate_financial_impact(
    excess_patient_days: int | None,
    excess_readmissions: int | None,
    cost_per_bed_day: float = 8000.0,
    cost_per_readmission: float = 25000.0,
) -> Dict[str, Any]:
    """
    Conservative financial impact estimator.
    """

    total_cost = 0.0
    components = {}

    if excess_patient_days:
        los_cost = excess_patient_days * cost_per_bed_day
        components["los_cost"] = round(los_cost, 2)
        total_cost += los_cost

    if excess_readmissions:
        readmit_cost = excess_readmissions * cost_per_readmission
        components["readmission_cost"] = round(readmit_cost, 2)
        total_cost += readmit_cost

    return {
        "total_annual_cost": round(total_cost, 2),
        "components": components,
    }


# =====================================================
# RISK ASSESSMENT
# =====================================================

def derive_risk_level(total_annual_cost: float | None) -> str:
    """
    Converts financial impact into executive risk band.
    """

    if total_annual_cost is None:
        return "LOW"

    if total_annual_cost >= 50_000_000:
        return "CRITICAL"
    if total_annual_cost >= 10_000_000:
        return "HIGH"
    if total_annual_cost >= 2_000_000:
        return "MEDIUM"

    return "LOW" 
