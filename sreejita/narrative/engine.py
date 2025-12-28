# sreejita/narrative/engine.py

from typing import Dict, Any

from .calculators import (
    calculate_excess_los_impact,
    calculate_readmission_impact,
    estimate_financial_impact,
    derive_risk_level,
)


# =====================================================
# NARRATIVE ENGINE (DETERMINISTIC)
# =====================================================

def generate_narrative(
    domain_result: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """
    Generates executive-grade narrative WITHOUT LLM.

    Returns structured business narrative ready for PDF.
    """

    kpis = domain_result.get("kpis", {})
    insights = domain_result.get("insights", [])

    # -------------------------------------------------
    # OPERATIONAL IMPACT
    # -------------------------------------------------
    los_impact = calculate_excess_los_impact(kpis)
    readmission_impact = calculate_readmission_impact(kpis)

    operational_points = []

    if los_impact:
        operational_points.append(
            f"Average length of stay exceeds target by "
            f"{los_impact['excess_days_per_patient']} days, resulting in "
            f"approximately {los_impact['excess_patient_days']:,} excess patient-days annually."
        )

    if readmission_impact:
        operational_points.append(
            f"Readmission rates exceed target by "
            f"{readmission_impact['excess_readmission_rate']} percentage points, "
            f"leading to an estimated {readmission_impact['excess_readmissions']:,} avoidable readmissions per year."
        )

    if not operational_points:
        operational_points.append(
            "Operational performance indicators suggest stable performance within expected thresholds."
        )

    operational_impact = " ".join(operational_points)

    # -------------------------------------------------
    # FINANCIAL IMPACT
    # -------------------------------------------------
    financial_calc = estimate_financial_impact(
        excess_patient_days=los_impact.get("excess_patient_days") if los_impact else None,
        excess_readmissions=readmission_impact.get("excess_readmissions") if readmission_impact else None,
    )

    if financial_calc["total_annual_cost"] > 0:
        financial_impact = (
            f"These inefficiencies may be contributing to approximately "
            f"₹{financial_calc['total_annual_cost']:,.0f} in avoidable annual costs."
        )
    else:
        financial_impact = (
            "No material financial leakage identified based on available performance indicators."
        )

    # -------------------------------------------------
    # RISK STATEMENT
    # -------------------------------------------------
    risk_level = derive_risk_level(financial_calc.get("total_annual_cost"))

    risk_statement = {
        "CRITICAL": "If left unaddressed, these issues pose a significant risk to financial sustainability and quality outcomes.",
        "HIGH": "Sustained underperformance may impact operational efficiency and reimbursement performance.",
        "MEDIUM": "Moderate operational risks exist and should be addressed through targeted interventions.",
        "LOW": "Current risk exposure appears manageable under existing controls.",
    }.get(risk_level, "Risk level undetermined.")

    # -------------------------------------------------
    # EXECUTIVE SUMMARY (1-MINUTE READ)
    # -------------------------------------------------
    executive_summary = (
        "Your facility is experiencing performance variances that affect operational efficiency "
        "and cost containment. Targeted interventions focused on discharge efficiency and "
        "care coordination can yield measurable improvements in both outcomes and cost."
    )

    # -------------------------------------------------
    # FINAL STRUCTURED OUTPUT
    # -------------------------------------------------
    return {
        "executive_summary": executive_summary,
        "operational_impact": operational_impact,
        "financial_impact": financial_impact,
        "risk_statement": risk_statement,
        "risk_level": risk_level,
    }
