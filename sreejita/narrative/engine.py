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
    - Safe for PDF / UI / API
    """

    # Defensive normalization
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []

    # =================================================
    # EXECUTIVE SUMMARY (REALITY-AWARE)
    # =================================================
    summary: List[str] = []

    # ---- Domain truth enforcement (Healthcare) ----
    if domain == "healthcare":
        long_stay_rate = kpis.get("long_stay_rate", 0) or 0
        readmission_rate = kpis.get("readmission_rate", 0) or 0

        if isinstance(long_stay_rate, (int, float)) and long_stay_rate > 0.25:
            summary.append(
                f"Operational strain detected: {long_stay_rate:.1%} of patients exceed "
                "length-of-stay targets, indicating discharge bottlenecks and reduced bed availability."
            )

        if isinstance(readmission_rate, (int, float)) and readmission_rate > 0.15:
            summary.append(
                f"Clinical risk elevated: Readmission rate at {readmission_rate:.1%} suggests "
                "gaps in discharge planning or follow-up care."
            )

    # ---- Insight reinforcement (secondary) ----
    for ins in insights[:2]:
        title = ins.get("title")
        so_what = ins.get("so_what")
        if title and so_what:
            summary.append(f"{title}: {so_what}")

    # ---- FIX A: De-duplicate executive summary ----
    seen = set()
    clean_summary: List[str] = []
    for s in summary:
        if s not in seen:
            clean_summary.append(s)
            seen.add(s)
    summary = clean_summary

    # ---- Absolute fallback (never empty) ----
    if not summary:
        summary.append(
            "Operational indicators are within expected thresholds with no immediate critical risks detected."
        )

    # =================================================
    # FINANCIAL IMPACT (SAFE & KPI-TIED)
    # =================================================
    financial: List[str] = []

    if domain == "healthcare":
        avg_cost = kpis.get("avg_cost_per_patient")
        avg_los = kpis.get("avg_los")

        # FIX B: strict numeric safety
        if (
            isinstance(avg_cost, (int, float))
            and isinstance(avg_los, (int, float))
            and avg_los > 7
        ):
            financial.append(
                f"Extended length of stay is increasing cost per patient "
                f"(average ${avg_cost:,.0f}), reducing throughput efficiency."
            )

    # Generic KPI-driven fallback
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
    # RISKS (CRITICAL + RISK + WARNING)
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

    # Prefer domain-generated recommendations
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

    # KPI-driven fallback (Healthcare)
    if not actions and domain == "healthcare":
        if isinstance(kpis.get("long_stay_rate"), (int, float)) and kpis["long_stay_rate"] > 0.15:
            actions.append(
                ActionItem(
                    action="Audit discharge planning for long-stay patients",
                    owner="Clinical Operations",
                    timeline="60 days",
                    success_kpi="Reduce long stay rate below 15%",
                )
            )

    # Absolute fallback (never empty)
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
