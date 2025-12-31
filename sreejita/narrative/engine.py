# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List

from .benchmarks import HEALTHCARE_BENCHMARKS as B


# =====================================================
# OUTPUT MODELS
# =====================================================

@dataclass
class ActionItem:
    action: str
    owner: str
    timeline: str
    success_kpi: str


@dataclass
class NarrativeResult:
    executive_summary: List[str]
    financial_impact: List[str]
    risks: List[str]
    action_plan: List[ActionItem]
    key_findings: List[Dict[str, Any]]


# =====================================================
# NARRATIVE ENGINE (DETERMINISTIC)
# =====================================================

def build_narrative(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> NarrativeResult:
    """
    Narrative Engine v6.3 — EXECUTIVE STABLE

    Design rules:
    - Max 5 executive summary bullets
    - No duplicate insight inflation
    - Deterministic risk detection
    - Zero dataset assumptions
    """

    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []

    summary: List[str] = []
    financial: List[str] = []
    risks: List[str] = []
    actions: List[ActionItem] = []

    calculated_savings: str | None = None

    # =================================================
    # HEALTHCARE DOMAIN LOGIC
    # =================================================
    if domain == "healthcare":

        # -------------------------------------------------
        # 1. GOVERNANCE CONTEXT
        # -------------------------------------------------
        interpretation = kpis.get("board_confidence_interpretation")
        if interpretation:
            summary.append(f"GOVERNANCE ALERT: {interpretation}")

        if kpis.get("dataset_shape") == "aggregated_operational":
            summary.append(
                "NOTICE: Assessment is based on aggregated operational data; conclusions reflect broad trends."
            )

        # -------------------------------------------------
        # 2. OPERATIONAL IMPACT & PROJECTION
        # -------------------------------------------------
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients")
        cost_per_day = kpis.get("avg_cost_per_day", 2000)

        if isinstance(avg_los, (int, float)) and avg_los > 0:
            target = B["avg_los"]["good"]

            if avg_los > target and isinstance(total_patients, (int, float)):
                excess_days = avg_los - target
                opportunity_loss = excess_days * cost_per_day * total_patients

                if opportunity_loss >= 1_000_000:
                    calculated_savings = f"${opportunity_loss / 1_000_000:.1f}M"
                else:
                    calculated_savings = f"${opportunity_loss / 1_000:.0f}K"

                blocked_beds = (excess_days * total_patients) / 365

                summary.append(
                    f"FINANCIAL IMPACT: Excess LOS represents a {calculated_savings} opportunity and blocks ~{blocked_beds:.0f} beds annually."
                )

                one_day_value = total_patients * cost_per_day
                one_day_str = (
                    f"${one_day_value / 1_000_000:.1f}M"
                    if one_day_value >= 1_000_000
                    else f"${one_day_value / 1_000:.0f}K"
                )
                financial.append(
                    f"Projection: Reducing average LOS by 1 day would recover approximately {one_day_str} annually."
                )

                if avg_los > B["avg_los"]["critical"]:
                    summary.append(
                        f"CRITICAL: Average LOS ({avg_los:.1f} days) exceeds crisis threshold."
                    )
                else:
                    summary.append(
                        f"Efficiency Gap: Average LOS ({avg_los:.1f} days) exceeds benchmark ({target} days)."
                    )

            else:
                summary.append(
                    f"Efficiency: LOS ({avg_los:.1f} days) performs within benchmark expectations."
                )

        # -------------------------------------------------
        # 3. CLINICAL JUSTIFICATION BRIDGE
        # -------------------------------------------------
        readm = kpis.get("readmission_rate")
        long_stay = kpis.get("long_stay_rate")

        if isinstance(readm, (int, float)):
            if readm > B["readmission_rate"]["critical"]:
                summary.append(f"Quality Alert: Readmission rate ({readm:.1%}) is elevated.")
                if isinstance(long_stay, (int, float)) and long_stay > 0.25:
                    summary.append(
                        "INEFFICIENCY SIGNAL: Extended LOS is not offset by improved outcomes."
                    )
            else:
                if isinstance(long_stay, (int, float)) and long_stay > 0.25:
                    summary.append(
                        "CLINICAL CONTEXT: Extended LOS may be clinically justified as outcomes remain controlled."
                    )
        else:
            risks.append(
                "Outcome justification unavailable; efficiency vs quality trade-offs cannot be validated."
            )

        # -------------------------------------------------
        # 4. FINANCIAL HEALTH
        # -------------------------------------------------
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost")

        if isinstance(avg_cost, (int, float)) and isinstance(benchmark_cost, (int, float)):
            if avg_cost > benchmark_cost:
                diff = avg_cost - benchmark_cost
                summary.append(
                    f"Financial Alert: Cost per patient exceeds benchmark by ${diff:,.0f}."
                )

        # -------------------------------------------------
        # 5. INSIGHT INTEGRATION (NON-DUPLICATIVE)
        # -------------------------------------------------
        for ins in insights:
            if ins.get("executive_summary_flag"):
                clean = ins.get("so_what", "").replace("<br/>", "; ")
                summary.append(f"KEY DRIVER: {clean}")

    # =================================================
    # UNIVERSAL FALLBACKS
    # =================================================
    if not summary:
        summary.append("Operational indicators are within expected thresholds.")

    if not financial:
        if isinstance(kpis.get("avg_cost_per_patient"), (int, float)):
            financial.append(
                f"Current cost structure is stable at ${kpis['avg_cost_per_patient']:,.0f} per patient."
            )
        else:
            financial.append("No material financial risks detected.")

    # =================================================
    # RISK SYNTHESIS
    # =================================================
    if any("CRITICAL" in s for s in summary):
        risks.append("Critical operational thresholds breached.")
    if any("GOVERNANCE" in s for s in summary):
        risks.append("Governance confidence degraded.")

    # =================================================
    # ACTION PLAN (TOP 5, IMPACT-AWARE)
    # =================================================
    for rec in recommendations[:5]:
        if not isinstance(rec, dict):
            continue

        action_text = rec.get("action", "Review performance metrics")
        outcome = rec.get("expected_outcome", "Improve KPI")
        timeline = rec.get("timeline", "")

        if calculated_savings and any(
            k in action_text.lower() for k in ("los", "discharge", "pathway")
        ):
            outcome = f"{outcome} (Est. Impact: {calculated_savings})"

        prefix = "⚡ [90-DAY FOCUS] " if any(
            x in timeline.lower() for x in ("30", "60", "90", "immediate", "q1")
        ) else ""

        actions.append(
            ActionItem(
                action=f"{prefix}{action_text}",
                owner=rec.get("owner", "Operations"),
                timeline=timeline,
                success_kpi=outcome,
            )
        )

    if not actions:
        actions.append(
            ActionItem(
                "Monitor key performance indicators",
                "Operations",
                "Ongoing",
                "Sustained stability",
            )
        )

    # =================================================
    # FINAL EXECUTIVE DISCIPLINE
    # =================================================
    summary = list(dict.fromkeys(summary))[:5]

    return NarrativeResult(
        executive_summary=summary,
        financial_impact=financial,
        risks=list(dict.fromkeys(risks)),
        action_plan=actions,
        key_findings=insights,
    )


def generate_narrative(*args, **kwargs):
    return build_narrative(*args, **kwargs)
