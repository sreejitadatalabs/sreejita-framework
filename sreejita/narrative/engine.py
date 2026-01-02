# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# =====================================================
# OUTPUT MODELS (STABLE)
# =====================================================

@dataclass
class ActionItem:
    action: str
    owner: str
    timeline: str
    expected_outcome: str


@dataclass
class NarrativeResult:
    executive_summary: List[str]
    key_insights: List[str]
    risks: List[str]
    action_plan: List[ActionItem]


# =====================================================
# NARRATIVE ENGINE (DETERMINISTIC, FORMAT-ONLY)
# =====================================================

def build_narrative(executive_payload: Dict[str, Any]) -> NarrativeResult:
    """
    Narrative Engine — FINAL UNIVERSAL VERSION

    RULES:
    - Does NOT compute intelligence
    - Does NOT read raw KPIs
    - Does NOT apply benchmarks
    - ONLY explains Executive Cognition output
    """

    if not isinstance(executive_payload, dict):
        return NarrativeResult([], [], [], [])

    # -------------------------------------------------
    # EXECUTIVE SUMMARY (1-MINUTE)
    # -------------------------------------------------
    summary: List[str] = []

    brief = executive_payload.get("executive_brief")
    if isinstance(brief, str) and brief.strip():
        summary.append(brief.strip())

    board = executive_payload.get("board_readiness", {})
    if board:
        summary.append(
            f"Board Readiness Score: {board.get('score','-')} / 100 "
            f"({board.get('band','Unknown')})."
        )

    # -------------------------------------------------
    # KEY INSIGHTS (STRUCTURED)
    # -------------------------------------------------
    key_insights: List[str] = []
    insight_block = executive_payload.get("insights", {})

    for tier in ("strengths", "warnings", "risks"):
        for ins in insight_block.get(tier, []):
            so_what = ins.get("so_what")
            if so_what:
                key_insights.append(so_what)

    # -------------------------------------------------
    # RISKS (EXECUTIVE SAFE)
    # -------------------------------------------------
    risks: List[str] = []

    for ins in insight_block.get("risks", []):
        title = ins.get("title")
        if title:
            risks.append(title)

    if not risks and board.get("band") in ("LOW", "MODERATE"):
        risks.append("Operational and governance risks require monitoring.")

    # -------------------------------------------------
    # ACTION PLAN (TOP 5 ONLY)
    # -------------------------------------------------
    actions: List[ActionItem] = []

    for rec in executive_payload.get("recommendations", [])[:5]:
        actions.append(
            ActionItem(
                action=rec.get("action", "Review performance"),
                owner=rec.get("owner", "Management"),
                timeline=rec.get("timeline", "TBD"),
                expected_outcome=rec.get("goal", "Improve outcomes"),
            )
        )

    if not actions:
        actions.append(
            ActionItem(
                action="Continue monitoring key indicators",
                owner="Leadership",
                timeline="Ongoing",
                expected_outcome="Sustained performance stability",
            )
        )

    # -------------------------------------------------
    # FINAL DISCIPLINE
    # -------------------------------------------------
    return NarrativeResult(
        executive_summary=summary[:3],     # keep tight
        key_insights=key_insights[:7],
        risks=risks[:5],
        action_plan=actions,
    )


# =====================================================
# BACKWARD-COMPATIBILITY ALIAS
# =====================================================

def generate_narrative(executive_payload: Dict[str, Any]):
    """
    Legacy alias — DO NOT USE for new logic.
    """
    return build_narrative(executive_payload)
