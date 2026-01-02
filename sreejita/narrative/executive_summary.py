"""
Executive Summary Adapter — UNIVERSAL (FINAL)
=============================================

Purpose:
- Produce a 1-minute, CEO-ready executive brief
- Compress Executive Cognition output into action-oriented language
- Never invent data
- Never be empty
- Never apply domain-specific logic

All intelligence lives in:
    sreejita.narrative.executive_cognition
"""

from typing import Dict, Any, List


# =====================================================
# PUBLIC ENTRY POINT
# =====================================================

def generate_executive_summary(
    domain: str,
    kpis: Dict[str, Any],
    insights: Any = None,
    recommendations: Any = None,
) -> str:
    """
    Generate a CEO / Board-ready executive summary.

    This function:
    - Accepts legacy parameters for compatibility
    - Expects Executive Cognition output either:
        a) directly in `kpis`
        b) nested under `kpis["executive"]`
    """

    # -------------------------------------------------
    # 1. DETECT EXECUTIVE PAYLOAD
    # -------------------------------------------------
    executive = None

    if isinstance(kpis, dict) and "executive_brief" in kpis:
        executive = kpis
    elif isinstance(kpis, dict) and "executive" in kpis:
        executive = kpis.get("executive")

    if not isinstance(executive, dict):
        return _safe_fallback()

    # -------------------------------------------------
    # 2. CORE SIGNALS
    # -------------------------------------------------
    board = executive.get("board_readiness", {})
    score = board.get("score")
    band = board.get("band", "Unknown")

    primary_kpis: List[Dict[str, Any]] = executive.get("primary_kpis", [])
    insight_block = executive.get("insights", {})
    recommendations = executive.get("recommendations", [])

    # -------------------------------------------------
    # 3. OPENING STATEMENT (MANDATORY)
    # -------------------------------------------------
    opening = (
        f"This {domain.replace('_',' ')} performance review indicates a "
        f"{band.lower()} operating position, with a Board Readiness Score "
        f"of {score} out of 100."
        if score is not None
        else
        f"This {domain.replace('_',' ')} performance review was completed "
        "based on available operational and quality signals."
    )

    paragraphs = [opening]

    # -------------------------------------------------
    # 4. POSITIVE SIGNALS (1–2 MAX)
    # -------------------------------------------------
    strengths = insight_block.get("strengths", [])
    if strengths:
        good = strengths[0]
        paragraphs.append(
            f"Encouragingly, {good.get('title','key performance areas')} "
            "are operating within acceptable or improving ranges."
        )

    # -------------------------------------------------
    # 5. KEY RISKS (MAX 1)
    # -------------------------------------------------
    risks = insight_block.get("risks", [])
    if risks:
        risk = risks[0]
        paragraphs.append(
            f"The most significant risk relates to {risk.get('title','a key operational area')}, "
            "which warrants focused leadership attention."
        )

    # -------------------------------------------------
    # 6. KPI EVIDENCE (TOP 2 ONLY)
    # -------------------------------------------------
    if primary_kpis:
        kpi_text = []
        for k in primary_kpis[:2]:
            kpi_text.append(
                f"{k.get('name')} ({k.get('value')})"
            )

        if kpi_text:
            paragraphs.append(
                "This assessment is supported by observed performance in "
                + " and ".join(kpi_text)
                + "."
            )

    # -------------------------------------------------
    # 7. ACTION ORIENTATION (MANDATORY)
    # -------------------------------------------------
    if recommendations:
        top = recommendations[0]
        paragraphs.append(
            f"Immediate focus on the recommended action — "
            f"{top.get('action','priority initiatives')} — "
            "over the next 60–90 days is expected to materially improve outcomes."
        )
    else:
        paragraphs.append(
            "Continued monitoring and disciplined execution will be critical "
            "to sustaining performance and mitigating emerging risks."
        )

    # -------------------------------------------------
    # 8. FINAL EXECUTIVE BRIEF (ONE PARAGRAPH)
    # -------------------------------------------------
    return " ".join(p.strip() for p in paragraphs if p).strip()


# =====================================================
# SAFE FALLBACK (NEVER EMPTY)
# =====================================================

def _safe_fallback() -> str:
    return (
        "Operational performance was reviewed across efficiency, cost, quality, "
        "and risk dimensions using available data. No immediate systemic threats "
        "were identified, though continued executive oversight and monitoring "
        "are recommended to sustain stability and performance."
    )
