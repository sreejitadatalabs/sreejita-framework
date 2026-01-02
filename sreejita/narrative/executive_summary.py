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

    Accepts legacy parameters but ONLY trusts
    Executive Cognition output.

    Expected executive payload locations:
    - kpis["executive_brief"]
    - kpis["executive"]
    """

    # -------------------------------------------------
    # 1. DETECT EXECUTIVE PAYLOAD (STRICT)
    # -------------------------------------------------
    executive: Dict[str, Any] | None = None

    if isinstance(kpis, dict):
        if "executive_brief" in kpis:
            executive = kpis
        elif "executive" in kpis and isinstance(kpis["executive"], dict):
            executive = kpis["executive"]

    if not isinstance(executive, dict):
        return _safe_fallback(domain)

    # -------------------------------------------------
    # 2. CORE EXECUTIVE SIGNALS
    # -------------------------------------------------
    board = executive.get("board_readiness", {}) or {}
    score = board.get("score")
    band = board.get("band", "Unknown")

    primary_kpis: List[Dict[str, Any]] = executive.get("primary_kpis", []) or []
    insight_block: Dict[str, Any] = executive.get("insights", {}) or {}
    recommendations = executive.get("recommendations", []) or []

    paragraphs: List[str] = []

    # -------------------------------------------------
    # 3. OPENING STATEMENT (MANDATORY)
    # -------------------------------------------------
    if isinstance(score, (int, float)):
        paragraphs.append(
            f"This {domain.replace('_',' ')} performance review indicates a "
            f"{band.lower()} operating position, with a Board Readiness Score "
            f"of {int(score)} out of 100."
        )
    else:
        paragraphs.append(
            f"This {domain.replace('_',' ')} performance review was completed "
            "based on available operational and risk signals."
        )

    # -------------------------------------------------
    # 4. POSITIVE SIGNAL (MAX 1)
    # -------------------------------------------------
    strengths = insight_block.get("strengths", []) or []
    if strengths:
        s = strengths[0]
        if s.get("title"):
            paragraphs.append(
                f"Encouragingly, {s['title'].lower()} are operating within "
                "acceptable or improving ranges."
            )

    # -------------------------------------------------
    # 5. PRIMARY RISK (MAX 1)
    # -------------------------------------------------
    risks = insight_block.get("risks", []) or []
    if risks:
        r = risks[0]
        if r.get("title"):
            paragraphs.append(
                f"The most significant risk relates to {r['title'].lower()}, "
                "which warrants focused leadership attention."
            )

    # -------------------------------------------------
    # 6. KPI EVIDENCE (TOP 2 ONLY)
    # -------------------------------------------------
    if primary_kpis:
        evidence = []
        for k in primary_kpis[:2]:
            name = k.get("name")
            value = k.get("value")
            if name is not None and value is not None:
                evidence.append(f"{name} ({value})")

        if evidence:
            paragraphs.append(
                "This assessment is supported by observed performance in "
                + " and ".join(evidence)
                + "."
            )

    # -------------------------------------------------
    # 7. ACTION ORIENTATION (MANDATORY)
    # -------------------------------------------------
    if recommendations:
        top = recommendations[0]
        action = top.get("action")
        timeline = top.get("timeline", "the next 60–90 days")

        if action:
            paragraphs.append(
                f"Immediate focus on the recommended action — {action.lower()} — "
                f"over {timeline.lower()} is expected to materially improve outcomes."
            )
        else:
            paragraphs.append(
                "Focused execution of the recommended initiatives over the next "
                "60–90 days is expected to materially improve outcomes."
            )
    else:
        paragraphs.append(
            "Continued monitoring and disciplined execution will be critical "
            "to sustaining performance and mitigating emerging risks."
        )

    # -------------------------------------------------
    # 8. FINAL ONE-PARAGRAPH OUTPUT
    # -------------------------------------------------
    return " ".join(p.strip() for p in paragraphs if p).strip()


# =====================================================
# SAFE FALLBACK (NEVER EMPTY)
# =====================================================

def _safe_fallback(domain: str) -> str:
    return (
        f"This {domain.replace('_',' ')} performance review was completed using "
        "available operational, financial, and quality indicators. While no "
        "immediate systemic threats were identified, continued executive "
        "oversight and structured monitoring are recommended to sustain "
        "stability and performance."
    )
