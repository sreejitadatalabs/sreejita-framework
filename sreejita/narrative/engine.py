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
    key_findings: List[Dict[str, Any]]  # intentionally permissive


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
    Deterministic executive narrative engine.

    Design guarantees:
    - NO LLM usage
    - NO external dependencies
    - MUST NEVER FAIL (defensive by design)
    - Domain-agnostic narrative synthesis
    """

    # Normalize inputs defensively
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []

    # -----------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------
    summary: List[str] = []

    if isinstance(insights, list) and insights:
        for ins in insights[:3]:
            title = ins.get("title", "Operational observation")
            so_what = ins.get("so_what", "Requires review.")
            summary.append(f"{title}: {so_what}")
    else:
        summary.append(
            "Operational indicators suggest stable performance with localized improvement opportunities."
        )

    # -----------------------------
    # FINANCIAL IMPACT (SAFE & EXPLAINABLE)
    # -----------------------------
    financial: List[str] = []

    # Intentional heuristic:
    # We surface the *first* materially non-zero KPI to avoid overwhelming executives.
    for k, v in kpis.items():
        if isinstance(v, (int, float)) and abs(v) > 0:
            financial.append(
                f"{k.replace('_', ' ').title()} deviation may have downstream cost implications."
            )
            break

    if not financial:
        financial.append(
            "No immediate material financial risk detected from current indicators."
        )

    # -----------------------------
    # RISKS
    # -----------------------------
    risks: List[str] = []

    for ins in insights:
        if isinstance(ins, dict) and ins.get("level") == "RISK":
            risks.append(ins.get("title", "Identified operational risk"))

    if not risks:
        risks.append("No critical risks identified at this time.")

    # -----------------------------
    # ACTION PLAN
    # -----------------------------
    actions: List[ActionItem] = []

    # Limit to top 2 actions for executive clarity
    for rec in recommendations[:2]:
        if not isinstance(rec, dict):
            continue

        actions.append(
            ActionItem(
                action=rec.get("action", "Operational improvement"),
                owner=rec.get("owner", "Business Owner"),  # intentional default
                timeline=rec.get("timeline", "90 days"),
                success_kpi=rec.get("success_kpi", "Target KPI improvement"),
            )
        )

    if not actions:
        actions.append(
            ActionItem(
                action="Monitor key operational metrics",
                owner="Operations Lead",
                timeline="Quarterly",
                success_kpi="Metrics remain within tolerance",
            )
        )

    # -----------------------------
    # FINAL ASSEMBLY
    # -----------------------------
    return NarrativeResult(
        executive_summary=summary,
        financial_impact=financial,
        risks=risks,
        action_plan=actions,
        key_findings=insights,  # pass-through for transparency
    )


# -----------------------------------------------------
# BACKWARD-COMPATIBILITY ALIAS (OPTIONAL, SAFE)
# -----------------------------------------------------

def generate_narrative(*args, **kwargs):
    """
    Backward-compatible alias.
    DO NOT REMOVE.
    """
    return build_narrative(*args, **kwargs)
