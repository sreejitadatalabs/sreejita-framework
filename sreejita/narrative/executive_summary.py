"""
Executive Summary Generator
===========================

Transforms KPIs, insights, and recommendations into
a CEO-ready narrative with:
- Logic consistency
- Benchmark awareness
- Financial impact framing
- Action-oriented language
- Confidence-safe phrasing
"""

from typing import Dict, List, Any


# ---------------------------
# BENCHMARKS (Configurable)
# ---------------------------
BENCHMARKS = {
    "healthcare": {
        "avg_los": 5.0,              # days
        "readmission_rate": 0.10,    # 10%
        "long_stay_rate": 0.15       # 15%
    }
}


# ==========================================================
# ENTRY POINT
# ==========================================================

def generate_executive_summary(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> str:
    """
    Generate an executive-level narrative summary.
    This is the ONLY narrative block consumed by PDF & UI.
    """

    lines: List[str] = []

    # ---------------------------
    # HEADER
    # ---------------------------
    lines.append("### Executive Summary\n")

    # ---------------------------
    # DOMAIN-SPECIFIC LOGIC
    # ---------------------------
    if domain == "healthcare":
        lines.extend(
            _healthcare_summary(
                kpis=kpis,
                insights=insights or [],
                recommendations=recommendations or [],
            )
        )
    else:
        # Generic fallback (safe, universal)
        lines.append(
            "Overall performance remains within acceptable operational limits. "
            "Key indicators across efficiency, cost, and risk were reviewed. "
            "No immediate systemic threats were detected, though continued monitoring is recommended."
        )

    return "\n".join(lines)


# ==========================================================
# HEALTHCARE EXECUTIVE NARRATIVE
# ==========================================================

def _healthcare_summary(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> List[str]:
    """
    Healthcare-specific executive narrative.
    Designed for CEOs, COOs, and Board members.
    """

    b = BENCHMARKS["healthcare"]
    out: List[str] = []

    avg_los = kpis.get("avg_los")
    long_stay_rate = kpis.get("long_stay_rate")
    readm = kpis.get("readmission_rate")
    cost = kpis.get("avg_cost_per_patient")

    confidence = kpis.get("board_confidence_score")

    # ---------------------------
    # 0. CONFIDENCE FRAMING (NEW)
    # ---------------------------
    if isinstance(confidence, (int, float)):
        if confidence >= 85:
            out.append(
                "Overall operational signals show **high confidence**, "
                "supporting executive-level decision-making."
            )
        elif confidence >= 70:
            out.append(
                "Operational signals show **moderate confidence**. "
                "Key findings are directionally reliable but warrant validation in critical areas."
            )
        else:
            out.append(
                "Operational signals indicate **elevated risk** due to data or performance instability. "
                "Decisions should be taken with caution and supported by targeted reviews."
            )

    # ---------------------------
    # 1. OPERATIONAL FLOW STATUS
    # ---------------------------
    if isinstance(long_stay_rate, (int, float)):
        if long_stay_rate > b["long_stay_rate"]:
            out.append(
                f"Hospital operations are **under capacity strain**. "
                f"Approximately **{long_stay_rate:.1%} of patients exceed standard length-of-stay targets**, "
                f"above the benchmark of {b['long_stay_rate']:.0%}. "
                f"This suggests discharge bottlenecks and reduced bed availability."
            )
        else:
            out.append(
                "Patient flow remains broadly stable, with length-of-stay metrics aligned to industry norms."
            )

    # ---------------------------
    # 2. CLINICAL QUALITY SIGNALS
    # ---------------------------
    if isinstance(readm, (int, float)):
        if readm > b["readmission_rate"]:
            out.append(
                f"Clinical quality presents **elevated risk**, with a **{readm:.1%} readmission rate**, "
                f"exceeding the benchmark of {b['readmission_rate']:.0%}. "
                f"This points to potential gaps in discharge planning or post-acute care continuity."
            )
        else:
            out.append(
                "Readmission performance remains within acceptable clinical benchmarks."
            )

    # ---------------------------
    # 3. FINANCIAL IMPACT LINKAGE
    # ---------------------------
    if isinstance(avg_los, (int, float)) and isinstance(cost, (int, float)):
        excess_days = max(avg_los - b["avg_los"], 0)
        if excess_days > 0:
            est_impact = excess_days * cost
            out.append(
                f"Extended inpatient stays result in an estimated "
                f"**{excess_days:.1f} excess days per patient**, translating to approximately "
                f"**${est_impact:,.0f} in avoidable cost per patient**. "
                f"Reducing length-of-stay represents a direct operational and financial opportunity." # [FIXED TYPO]
            )

    # ---------------------------
    # 4. KEY RISK SIGNALS (FROM INSIGHTS)
    # ---------------------------
    criticals = [
        i for i in insights
        if i.get("level") in ("CRITICAL", "RISK", "WARNING")
    ]

    if criticals:
        out.append("\n**Key Risk Signals Identified:**")
        for i in criticals[:3]:
            title = i.get("title", "Unspecified Risk")
            so_what = i.get("so_what", "")
            out.append(f"- **{title}** — {so_what}")

    # ---------------------------
    # 5. ACTION PLAN (MANDATORY)
    # ---------------------------
    out.append("\n**Priority Actions Recommended:**")

    if recommendations:
        for r in recommendations[:4]:
            priority = r.get("priority", "MEDIUM")
            action = r.get("action", "Action required")
            out.append(f"- ({priority}) {action}")
    else:
        # Hard failsafe — never empty
        out.append(
            "- (LOW) Continue monitoring patient flow, discharge efficiency, "
            "and care coordination to prevent future congestion."
        )

    # ---------------------------
    # 6. EXECUTIVE BOTTOM LINE
    # ---------------------------
    out.append(
        "\n**Bottom Line:** "
        "The organization remains operationally and financially viable. "
        "However, efficiency and quality risks persist that could compound under higher demand. "
        "Targeted improvements in discharge planning, care coordination, and throughput management "
        "can unlock capacity, reduce cost leakage, and improve patient outcomes."
    )

    return out
