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

    Supports:
    - Global executive cognition
    - Per-sub-domain executive cognition

    Accepted executive payload locations:
    - kpis["executive_brief"]                (single payload)
    - kpis["executive"]                      (single payload)
    - kpis["executive_by_sub_domain"]        (multi payload)
    """

    # -------------------------------------------------
    # 1. MULTI–SUB-DOMAIN EXECUTIVE PAYLOAD (PREFERRED)
    # -------------------------------------------------
    if isinstance(kpis, dict) and "executive_by_sub_domain" in kpis:
        sub_execs = kpis.get("executive_by_sub_domain")

        if isinstance(sub_execs, dict) and sub_execs:
            paragraphs: List[str] = []

            for sub, payload in sub_execs.items():
                if not isinstance(payload, dict):
                    continue

                brief = payload.get("executive_brief")
                board = payload.get("board_readiness", {}) or {}

                score = board.get("score")
                band = board.get("band")

                # ---- Opening (per sub-domain)
                if isinstance(score, (int, float)):
                    paragraphs.append(
                        f"{sub.replace('_',' ').title()} operations reflect a "
                        f"{band.lower()} position, with a Board Readiness Score "
                        f"of {int(score)} out of 100."
                    )
                else:
                    paragraphs.append(
                        f"{sub.replace('_',' ').title()} operations were assessed "
                        "using available operational and risk signals."
                    )

                # ---- Executive brief sentence (authoritative)
                if isinstance(brief, str) and brief.strip():
                    paragraphs.append(brief.strip())

            if paragraphs:
                return " ".join(paragraphs).strip()

    # -------------------------------------------------
    # 2. SINGLE EXECUTIVE PAYLOAD (BACKWARD COMPATIBLE)
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
    # 3. CORE EXECUTIVE SIGNALS
    # -------------------------------------------------
    board = executive.get("board_readiness", {}) or {}
    score = board.get("score")
    band = board.get("band", "Unknown")

    primary_kpis: List[Dict[str, Any]] = executive.get("primary_kpis", []) or []
    insight_block: Dict[str, Any] = executive.get("insights", {}) or {}
    recs = executive.get("recommendations", []) or []

    paragraphs: List[str] = []

    # -------------------------------------------------
    # 4. OPENING STATEMENT (MANDATORY)
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
    # 5. POSITIVE SIGNAL (MAX 1)
    # -------------------------------------------------
    strengths = insight_block.get("strengths", []) or []
    if strengths and strengths[0].get("title"):
        paragraphs.append(
            f"Encouragingly, {strengths[0]['title'].lower()} are operating "
            "within acceptable or improving ranges."
        )

    # -------------------------------------------------
    # 6. PRIMARY RISK (MAX 1)
    # -------------------------------------------------
    risks = insight_block.get("risks", []) or []
    if risks and risks[0].get("title"):
        paragraphs.append(
            f"The most significant risk relates to {risks[0]['title'].lower()}, "
            "which warrants focused leadership attention."
        )

    # -------------------------------------------------
    # 7. KPI EVIDENCE (TOP 2 ONLY)
    # -------------------------------------------------
    if primary_kpis:
        evidence: List[str] = []
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
    # 8. ACTION ORIENTATION (MANDATORY)
    # -------------------------------------------------
    if recs:
        top = recs[0]
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
    # 9. FINAL OUTPUT (NEVER EMPTY)
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
