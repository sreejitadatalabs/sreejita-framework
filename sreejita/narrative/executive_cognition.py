# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (FINAL)
# Sreejita Framework
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability


# =====================================================
# RISK BANDS (EXECUTIVE SAFE)
# =====================================================

EXECUTIVE_RISK_BANDS = [
    (85, "LOW", "🟢"),
    (70, "MEDIUM", "🟡"),
    (50, "HIGH", "🟠"),
    (0,  "CRITICAL", "🔴"),
]


def derive_risk_level(score: int) -> Dict[str, Any]:
    score = int(score or 0)

    for threshold, label, icon in EXECUTIVE_RISK_BANDS:
        if score >= threshold:
            return {
                "label": label,
                "icon": icon,
                "score": score,
                "display": f"{icon} {label} (Score: {score} / 100)",
            }

    return {
        "label": "CRITICAL",
        "icon": "🔴",
        "score": score,
        "display": f"🔴 CRITICAL (Score: {score} / 100)",
    }


# =====================================================
# KPI SELECTION (MAX 9, CONFIDENCE-DRIVEN)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap_map = kpis.get("_kpi_capabilities", {})
    conf_map = kpis.get("_confidence", {})

    ranked: List[Dict[str, Any]] = []

    for key, cap in cap_map.items():
        value = kpis.get(key)

        if not isinstance(value, (int, float)):
            continue

        confidence = conf_map.get(key, 0.6)

        # Capability importance weights (NOT magnitude-based)
        cap_weight = {
            Capability.QUALITY.value: 1.3,
            Capability.TIME_FLOW.value: 1.2,
            Capability.COST.value: 1.1,
            Capability.VOLUME.value: 1.0,
            Capability.VARIANCE.value: 1.0,
            Capability.ACCESS.value: 1.0,
        }.get(cap, 1.0)

        score = confidence * cap_weight

        ranked.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "value": round(value, 2),
            "capability": cap,
            "confidence": round(confidence, 2),
            "score": round(score, 3),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:9]   # 🔒 HARD MAX


# =====================================================
# INSIGHT STRUCTURING (EXECUTIVE LOGIC)
# =====================================================

def structure_insights(
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    strengths = [i for i in insights if i.get("level") == "STRENGTH"][:2]
    warnings = [i for i in insights if i.get("level") == "WARNING"][:2]
    risks = [i for i in insights if i.get("level") == "RISK"][:1]

    avg_conf = round(
        sum(i.get("confidence", 0.7) for i in insights) / max(len(insights), 1),
        2,
    )

    composite = {
        "title": "Overall Executive Assessment",
        "summary": (
            "Performance shows identifiable strengths, with "
            "manageable risk areas requiring focused leadership attention."
        ),
        "confidence": avg_conf,
    }

    return {
        "strengths": strengths,
        "warnings": warnings,
        "risks": risks,
        "composite": composite,
    }


# =====================================================
# BOARD READINESS SCORE (UNIVERSAL & HONEST)
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    confidence_map = kpis.get("_confidence", {})
    confident_kpis = [
        v for v in confidence_map.values()
        if isinstance(v, (int, float)) and v >= 0.6
    ]

    # Evidence strength (0–45)
    evidence_score = (len(confident_kpis) / 9) * 45

    # Coverage (0–25)
    coverage_keys = ["record_count", "data_completeness", "time_coverage_days"]
    coverage = sum(1 for k in coverage_keys if k in kpis and kpis[k] is not None)
    coverage_score = (coverage / len(coverage_keys)) * 25

    # Risk penalty
    risk_penalty = sum(
        10 for i in insights if i.get("level") == "RISK"
    )

    score = round(
        max(0, min(100, evidence_score + coverage_score + 30 - risk_penalty))
    )

    # Hard cap for weak data
    if kpis.get("data_completeness", 1) < 0.7:
        score = min(score, 60)

    return {
        "score": score,
        "band": (
            "BOARD READY" if score >= 85 else
            "REVIEW WITH CAUTION" if score >= 70 else
            "MANAGEMENT ONLY" if score >= 50 else
            "NOT BOARD SAFE"
        ),
    }


# =====================================================
# 1-MINUTE EXECUTIVE BRIEF (FINAL)
# =====================================================

def build_executive_brief(
    board_score: int,
    insight_block: Dict[str, Any],
    sub_domain: str,
) -> str:

    risk = derive_risk_level(board_score)

    brief = [
        f"This {sub_domain.replace('_',' ')} performance review is assessed as "
        f"{risk['label'].lower()}, with a Board Readiness Score of "
        f"{risk['score']} out of 100."
    ]

    if insight_block["strengths"]:
        brief.append(
            f"Positive signals include "
            f"{insight_block['strengths'][0]['title'].lower()}."
        )

    if insight_block["risks"]:
        brief.append(
            f"The primary risk relates to "
            f"{insight_block['risks'][0]['title'].lower()}, "
            "which requires timely leadership action."
        )

    brief.append(
        "Focused execution of the recommended actions over the next "
        "60–90 days can materially improve outcomes."
    )

    return " ".join(brief)


# =====================================================
# RECOMMENDATION NORMALIZATION
# =====================================================

def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    normalized = []

    for r in recommendations[:5]:
        normalized.append({
            "priority": r.get("priority", "MEDIUM"),
            "action": r.get("action"),
            "owner": r.get("owner"),
            "timeline": r.get("timeline"),
            "goal": r.get("goal"),
            "confidence": round(r.get("confidence", 0.7), 2),
        })

    return normalized


# =====================================================
# EXECUTIVE PAYLOAD (AUTHORITATIVE OUTPUT)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    primary_sub = kpis.get("primary_sub_domain", "unknown")

    primary_kpis = select_executive_kpis(kpis)
    insight_block = structure_insights(insights)
    board = compute_board_readiness_score(kpis, insights)

    executive_brief = build_executive_brief(
        board_score=board["score"],
        insight_block=insight_block,
        sub_domain=primary_sub,
    )

    executive_recs = normalize_recommendations(recommendations)

    return {
        "executive_brief": executive_brief,
        "primary_kpis": primary_kpis,
        "insights": insight_block,
        "recommendations": executive_recs,
        "board_readiness": board,
        "sub_domain": primary_sub,
    }
