# sreejita/narrative/rules.py
from typing import Dict, List
from .schema import ExecutiveFinding, ActionItem


def healthcare_rules(kpis: Dict) -> Dict:
    findings = []
    financial = []
    risks = []
    actions = []

    avg_los = kpis.get("avg_length_of_stay")
    readmission = kpis.get("readmission_rate")
    cost = kpis.get("avg_treatment_cost")

    # ---- LOS RULE ----
    if avg_los and avg_los > 5:
        excess_days = (avg_los - 5) * 1000  # conservative proxy
        findings.append(
            ExecutiveFinding(
                title="Prolonged Length of Stay",
                explanation="Patients are staying longer than the operational target.",
                impact=f"Estimated {excess_days:.0f} excess patient-days annually."
            )
        )
        financial.append(
            f"Excess LOS is likely increasing operating costs through higher bed occupancy and staffing."
        )
        risks.append(
            "Sustained capacity strain may impact admission throughput and patient satisfaction."
        )
        actions.append(
            ActionItem(
                action="Implement provider-level LOS review and discharge optimization",
                owner="Chief Medical Officer",
                timeline="90 days",
                success_kpi="Reduce average LOS to ≤5 days"
            )
        )

    # ---- READMISSION RULE ----
    if readmission and readmission > 0.1:
        findings.append(
            ExecutiveFinding(
                title="Elevated Readmissions",
                explanation="Readmission rate exceeds acceptable clinical threshold.",
                impact="Indicates care transition gaps or follow-up deficiencies."
            )
        )
        financial.append(
            "Higher readmissions may lead to avoidable treatment costs and payer penalties."
        )
        risks.append(
            "Regulatory exposure and quality score deterioration."
        )
        actions.append(
            ActionItem(
                action="Strengthen discharge planning and post-care follow-ups",
                owner="Clinical Operations",
                timeline="6 months",
                success_kpi="Reduce readmissions to ≤10%"
            )
        )

    return {
        "findings": findings,
        "financial": financial,
        "risks": risks,
        "actions": actions,
    }
