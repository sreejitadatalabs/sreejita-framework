# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (SREEJITA FRAMEWORK)
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability

# =====================================================
# EXECUTIVE RISK MODEL
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
# KPI SELECTION (EXECUTIVE SAFE, MAX 9)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap_map = kpis.get("_kpi_capabilities", {})
    conf_map = kpis.get("_confidence", {})

    scored: List[Dict[str, Any]] = []

    for key, cap in cap_map.items():
        value = kpis.get(key)

        if not isinstance(value, (int, float)):
            continue

        confidence = conf_map.get(key, 0.7)

        weight = {
            Capability.VOLUME.value: 1.0,
            Capability.TIME_FLOW.value: 1.2,
            Capability.COST.value: 1.1,
            Capability.QUALITY.value: 1.3,
            Capability.VARIANCE.value: 1.0,
            Capability.ACCESS.value: 1.0,
        }.get(cap, 1.0)

        score = abs(value) * confidence * weight

        scored.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "value": round(value, 2),
            "capability": cap,
            "confidence": round(confidence, 2),
            "score": round(score, 2),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:9]   # 🔒 HARD MAX


# =====================================================
# INSIGHT STRUCTURING (EXECUTIVE LOGIC)
# =====================================================

def structure_insights(
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    strengths = [i for i in insights if i.get("level") == "STRENGTH"][:2]
    warnings = [i for i in insights if i.get("level") == "WARNING"][:2]
    risks = [i for i in insights if i.get("level") == "RISK"][:1]

    composite_conf = round(
        sum(i.get("confidence", 0.7) for i in insights) / max(len(insights), 1),
        2,
    )

    composite = {
        "title": "Overall Executive Assessment",
        "summary": (
            "Performance shows identifiable strengths, with manageable "
            "risk areas requiring focused operational attention."
        ),
        "confidence": composite_conf,
    }

    return {
        "strengths": strengths,
        "warnings": warnings,
        "risks": risks,
        "composite": composite,
    }


# =====================================================
# 1-MINUTE EXECUTIVE BRIEF
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
            f"Key strengths include "
            f"{insight_block['strengths'][0]['title'].lower()}."
        )

    if insight_block["risks"]:
        brief.append(
            f"The primary risk relates to "
            f"{insight_block['risks'][0]['title'].lower()}, "
            "which requires timely leadership action."
        )

    brief.append(
        "Overall, leadership intervention at this stage can materially "
        "improve outcomes over the next operational cycle."
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
# BOARD READINESS SCORE (UNIVERSAL)
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    confidence_map = kpis.get("_confidence", {})
    avg_conf = sum(confidence_map.values()) / max(len(confidence_map), 1)

    kpi_signal = avg_conf * 45

    coverage_keys = [
        "total_volume",
        "data_completeness",
    ]
    coverage = sum(1 for k in coverage_keys if k in kpis)
    coverage_score = (coverage / len(coverage_keys)) * 25

    risk_penalty = sum(
        10 for i in insights if i.get("level") == "RISK"
    )

    final_score = round(
        max(0, min(100, kpi_signal + coverage_score + 30 - risk_penalty))
    )

    if kpis.get("data_completeness", 1) < 0.7:
        final_score = min(final_score, 60)

    return {
        "score": final_score,
        "band": (
            "BOARD READY" if final_score >= 85 else
            "REVIEW WITH CAUTION" if final_score >= 70 else
            "MANAGEMENT ONLY" if final_score >= 50 else
            "NOT BOARD SAFE"
        ),
    }


# =====================================================
# EXECUTIVE PAYLOAD (AUTHORITATIVE)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    # 1. KPI SELECTION
    primary_kpis = select_executive_kpis(kpis)

    # 2. INSIGHT STRUCTURE
    insight_block = structure_insights(insights)

    # 3. BOARD READINESS
    board = compute_board_readiness_score(kpis, insights)

    # 4. EXECUTIVE BRIEF
    executive_brief = build_executive_brief(
        board_score=board["score"],
        insight_block=insight_block,
        sub_domain=kpis.get("primary_sub_domain", "healthcare"),
    )

    # 5. RECOMMENDATIONS
    executive_recs = normalize_recommendations(recommendations)

    return {
        "executive_brief": executive_brief,
        "primary_kpis": primary_kpis,
        "insights": insight_block,
        "recommendations": executive_recs,
        "board_readiness": board,
        "sub_domain": kpis.get("primary_sub_domain"),
    }
