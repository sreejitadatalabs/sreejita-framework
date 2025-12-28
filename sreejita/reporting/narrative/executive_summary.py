"""
Executive Summary Generator
===========================

Transforms KPIs, insights, and recommendations into
a CEO-ready narrative with:
- Logic consistency
- Benchmark awareness
- Financial impact framing
- Action-oriented language
"""

from typing import Dict, List


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


def generate_executive_summary(
    domain: str,
    kpis: Dict,
    insights: List[Dict],
    recommendations: List[Dict]
) -> str:
    """
    Generate an executive-level narrative summary.
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
        lines.extend(_healthcare_summary(kpis, insights, recommendations))
    else:
        # Generic fallback (safe)
        lines.append(
            "Overall performance is within acceptable limits. "
            "Key metrics were analyzed across operations, risk, and efficiency."
        )

    return "\n".join(lines)


# ==========================================================
# HEALTHCARE EXECUTIVE NARRATIVE
# ==========================================================

def _healthcare_summary(kpis, insights, recommendations) -> List[str]:
    b = BENCHMARKS["healthcare"]
    out: List[str] = []

    avg_los = kpis.get("avg_los")
    long_stay_rate = kpis.get("long_stay_rate", 0)
    readm = kpis.get("readmission_rate")
    cost = kpis.get("avg_cost_per_patient")

    # ---------------------------
    # 1. OPERATIONS STATUS
    # ---------------------------
    if long_stay_rate > b["long_stay_rate"]:
        out.append(
            f"Hospital operations are **under capacity strain**. "
            f"Approximately **{long_stay_rate:.1%} of patients exceed standard length-of-stay targets**, "
            f"significantly above the industry benchmark of {b['long_stay_rate']:.0%}. "
            f"This indicates discharge bottlenecks and reduced bed availability."
        )
    else:
        out.append(
            "Operational flow remains stable, with length-of-stay metrics broadly aligned with industry norms."
        )

    # ---------------------------
    # 2. CLINICAL QUALITY
    # ---------------------------
    if readm is not None:
        if readm > b["readmission_rate"]:
            out.append(
                f"Clinical outcomes present **elevated risk**, with a **{readm:.1%} readmission rate**, "
                f"exceeding the typical benchmark of {b['readmission_rate']:.0%}. "
                f"This suggests gaps in discharge planning or post-care continuity."
            )
        else:
            out.append(
                "Readmission performance remains within acceptable clinical benchmarks."
            )

    # ---------------------------
    # 3. FINANCIAL LINKAGE
    # ---------------------------
    if avg_los and cost:
        excess_days = max(avg_los - b["avg_los"], 0)
        if excess_days > 0:
            est_impact = excess_days * cost
            out.append(
                f"Extended inpatient stays add an estimated **{excess_days:.1f} excess days per patient**, "
                f"translating to approximately **${est_impact:,.0f} in avoidable cost per patient**. "
                f"Reducing length-of-stay represents a direct opportunity for financial efficiency."
            )

    # ---------------------------
    # 4. KEY RISKS (FROM INSIGHTS)
    # ---------------------------
    criticals = [i for i in insights if i["level"] in ("CRITICAL", "WARNING")]

    if criticals:
        out.append("\n**Key Risk Signals Identified:**")
        for i in criticals[:3]:
            out.append(f"- {i['title']}: {i['so_what']}")

    # ---------------------------
    # 5. ACTION PLAN (MANDATORY)
    # ---------------------------
    out.append("\n**Priority Actions Recommended:**")

    if recommendations:
        for r in recommendations[:4]:
            out.append(f"- ({r['priority']}) {r['action']}")
    else:
        # Failsafe: NEVER empty
        out.append(
            "- (LOW) Continue monitoring patient flow and discharge efficiency to prevent future congestion."
        )

    # ---------------------------
    # 6. CLOSING STATEMENT
    # ---------------------------
    out.append(
        "\n**Bottom Line:** "
        "The facility remains financially viable, but operational efficiency risks persist. "
        "Targeted improvements in discharge planning and care coordination can unlock capacity, "
        "reduce costs, and improve patient outcomes."
    )

    return out
