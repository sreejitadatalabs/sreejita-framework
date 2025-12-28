# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List


# =====================================================
# NARRATIVE OUTPUT MODELS
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
    key_findings: List[Dict[str, Any]]  # pass-through (auditable)


# =====================================================
# PUBLIC API (🔥 DO NOT BREAK THIS)
# =====================================================

def build_narrative(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> NarrativeResult:
    """
    Deterministic executive narrative engine (v3.6.1).

    GUARANTEES:
    - No LLM usage
    - KPI-driven executive logic
    - Never empty sections
    - Domain-aware but not domain-coupled
    """

    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []

    # =================================================
    # EXECUTIVE SUMMARY (FIX 4 — REALITY AWARE)
    # =================================================
    summary: List[str] = []

    # ---- Healthcare-specific truth enforcement ----
    if domain == "healthcare":
        long_stay_rate = kpis.get("long_stay_rate", 0)
        readmission_rate = kpis.get("readmission_rate", 0)

        if long_stay_rate > 0.25:
            summary.append(
                f"Operational strain detected: {long_stay_rate:.1%} of patients exceed "
                "length-of-stay targets, indicating discharge bottlenecks and reduced bed availability."
            )

        if readmission_rate > 0.15:
            summary.append(
                f"Clinical risk elevated: Readmission rate at {readmission_rate:.1%} suggests "
                "gaps in discharge planning or follow-up care."
            )

    # ---- Insight-driven reinforcement (secondary) ----
    for ins in insights[:2]:
        title = ins.get("title")
        so_what = ins.get("so_what")
        if title and so_what:
            summary.append(f"{title}: {so_what}")

    # ---- Absolute fallback (never empty) ----
    if not summary:
        summary.append(
            "Operational indicators are within expected thresholds with no immediate critical risks detected."
        )

    # =================================================
    # FINANCIAL IMPACT (EXPLAINABLE & TIED TO KPIs)
    # =================================================
    financial: List[str] = []

    if domain == "healthcare":
        avg_cost = kpis.get("avg_cost_per_patient")
        avg_los = kpis.get("avg_los")

        if avg_cost and avg_los and avg_los > 7:
            financial.append(
                f"Extended length of stay is increasing cost per patient "
                f"(average ${avg_cost:,.0f}), reducing throughput efficiency."
            )

    if not financial:
        for k, v in kpis.items():
            if isinstance(v, (int, float)) and abs(v) > 0:
                financial.append(
                    f"{k.replace('_', ' ').title()} levels may have downstream financial implications."
                )
                break

    if not financial:
        financial.append("No immediate material financial risk detected.")

    # =================================================
    # RISKS (FIX — INCLUDE WARNING & CRITICAL)
    # =================================================
    risks: List[str] = []

    for ins in insights:
        if ins.get("level") in {"CRITICAL", "RISK", "WARNING"}:
            risks.append(ins.get("title", "Identified operational risk"))

    if not risks:
        risks.append("No critical or emerging risks identified at this time.")

    # =================================================
    # ACTION PLAN (EXECUTIVE-GRADE)
    # =================================================
    actions: List[ActionItem] = []

    # Prefer domain recommendations
    for rec in recommendations[:2]:
        if isinstance(rec, dict):
            actions.append(
                ActionItem(
                    action=rec.get("action", "Operational improvement"),
                    owner=rec.get("owner", "Operations Leadership"),
                    timeline=rec.get("timeline", "60–90 days"),
                    success_kpi=rec.get("success_kpi", "Primary KPI improvement"),
                )
            )

    # KPI-driven fallback actions (healthcare)
    if not actions and domain == "healthcare":
        if kpis.get("long_stay_rate", 0) > 0.15:
            actions.append(
                ActionItem(
                    action="Audit discharge planning for long-stay patients",
                    owner="Clinical Operations",
                    timeline="60 days",
                    success_kpi="Reduce long stay rate below 15%",
                )
            )

    if not actions:
        actions.append(
            ActionItem(
                action="Continue monitoring key operational metrics",
                owner="Operations Lead",
                timeline="Quarterly",
                success_kpi="Metrics remain within tolerance",
            )
        )

    # =================================================
    # FINAL ASSEMBLY
    # =================================================
    return NarrativeResult(
        executive_summary=summary,
        financial_impact=financial,
        risks=risks,
        action_plan=actions,
        key_findings=insights,  # transparent pass-through
    )


# -----------------------------------------------------
# BACKWARD COMPATIBILITY (DO NOT REMOVE)
# -----------------------------------------------------

def generate_narrative(*args, **kwargs):
    return build_narrative(*args, **kwargs)
