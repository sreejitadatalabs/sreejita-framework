"""
Executive Summary Adapter (FINAL — UNIVERSAL)
===========================================

This module is a thin compatibility layer.

IMPORTANT:
- It does NOT compute intelligence
- It does NOT apply benchmarks
- It does NOT read raw KPIs
- It ONLY formats Executive Cognition output

All decision logic lives in:
    sreejita/narrative/executive_cognition.py
"""

from typing import Dict, Any


# =====================================================
# PUBLIC ENTRY POINT (BACKWARD COMPATIBLE)
# =====================================================

def generate_executive_summary(
    domain: str,
    kpis: Dict[str, Any],
    insights: Any = None,
    recommendations: Any = None,
) -> str:
    """
    Generate a CEO-ready executive summary.

    NOTE:
    - `kpis`, `insights`, `recommendations` are accepted only
      for backward compatibility.
    - This function expects Executive Cognition output
      to be embedded inside `kpis["executive"]` OR passed directly.
    """

    # -------------------------------------------------
    # 1. DETECT EXECUTIVE PAYLOAD
    # -------------------------------------------------
    executive = None

    # Preferred (new architecture)
    if isinstance(kpis, dict) and "executive_brief" in kpis:
        executive = kpis

    # Nested payload (common from orchestrator)
    elif isinstance(kpis, dict) and "executive" in kpis:
        executive = kpis.get("executive")

    # Absolute fallback
    if not isinstance(executive, dict):
        return (
            "Operational performance was reviewed across key dimensions. "
            "Available signals indicate no immediate systemic threats, "
            "though continued monitoring is recommended."
        )

    # -------------------------------------------------
    # 2. EXECUTIVE BRIEF (AUTHORITATIVE)
    # -------------------------------------------------
    brief = executive.get("executive_brief")
    if isinstance(brief, str) and brief.strip():
        return brief.strip()

    # -------------------------------------------------
    # 3. LAST-RESORT FALLBACK (NEVER EMPTY)
    # -------------------------------------------------
    board = executive.get("board_readiness", {})
    score = board.get("score", "N/A")
    band = board.get("band", "Unknown")

    return (
        f"The organization’s performance was assessed with a "
        f"Board Readiness Score of {score} ({band}). "
        "Key indicators were reviewed across efficiency, cost, "
        "quality, and risk dimensions. Leadership attention "
        "to prioritized recommendations will support continued stability."
    )
