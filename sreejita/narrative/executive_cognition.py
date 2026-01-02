# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (FINAL)
# Sreejita Framework
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability


# =====================================================
# EXECUTIVE RISK BANDS (SAFE, BOARD-LEGIBLE)
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
                "display": f"{icon} {label} (Score: {score}/100)",
            }

    return {
        "label": "CRITICAL",
        "icon": "🔴",
        "score": score,
        "display": f"🔴 CRITICAL (Score: {score}/100)",
    }


# =====================================================
# EXECUTIVE KPI SELECTION (MAX 9, CAPABILITY-AWARE)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap_map = kpis.get("_kpi_capabilities", {}) or {}
    conf_map = kpis.get("_confidence", {}) or {}

    ranked: List[Dict[str, Any]] = []

    for key, capability in cap_map.items():
        value = kpis.get(key)

        if not isinstance(value, (int, float)):
            continue

        confidence = float(conf_map.get(key, 0.6))

        capability_weight = {
            Capability.QUALITY.value: 1.30,
            Capability.TIME_FLOW.value: 1.20,
            Capability.COST.value: 1.10,
            Capability.VOLUME.value: 1.00,
            Capability.VARIANCE.value: 1.00,
            Capability.ACCESS.value: 1.00,
        }.get(capability, 1.0)

        score = confidence * capability_weight

        ranked.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "value": round(value, 2),
            "capability": capability,
            "confidence": round(confidence, 2),
            "rank_score": round(score, 3),
        })

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)

    # 🔒 HARD RULE: max 9 KPIs
    return ranked[:9]


# =====================================================
# INSIGHT STRUCTURING (EXECUTIVE FLOW)
# =====================================================

def structure_insights(insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    strengths = [i for i in insights if i.get("level") == "STRENGTH"][:2]
    warnings  = [i for i in insights if i.get("level") == "WARNING"][:2]
    risks     = [i for i in insights if i.get("level") == "RISK"][:1]

    avg_conf = round(
        sum(i.get("confidence", 0.7) for i in insights) / max(len(insights), 1),
        2,
    )

    composite = {
        "title": "Overall Executive Assessment",
        "summary": (
            "Operational performance demonstrates identifiable strengths, "
            "with targeted risks requiring focused leadership action."
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
# BOARD READINESS SCORE (HONEST & GOVERNED)
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    conf_map = kpis.get("_confidence", {}) or {}
    confident_kpis = [
        v for v in conf_map.values()
        if isinstance(v, (int, float)) and v >= 0.6
    ]

    # Evidence strength (0–45)
    evidence_score = (len(confident_kpis) / 9) * 45

    # Coverage score (0–25)
    coverage_keys = ["total_volume", "data_completeness"]
    coverage_hits = sum(
        1 for k in coverage_keys
        if isinstance(kpis.get(k), (int, float))
    )
    coverage_score = (coverage_hits / len(coverage_keys)) * 25

    # Risk penalty
    risk_penalty = sum(10 for i in insights if i.get("level") == "RISK")

    score = round(
        max(0, min(100, evidence_score + coverage_score + 30 - risk_penalty))
    )

    # Hard safety cap for weak data
    if isinstance(kpis.get("data_completeness"), (int, float)) and kpis["data_completeness"] < 0.7:
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
# 1-MINUTE EXECUTIVE BRIEF (CEO LEGIBLE)
# =====================================================

def build_executive_brief(
    board_score: int,
    insight_block: Dict[str, Any],
    sub_domain: str,
) -> str:

    risk = derive_risk_level(board_score)

    brief: List[str] = []

    brief.append(
        f"This {sub_domain.replace('_', ' ')} performance review is assessed as "
        f"{risk['label'].lower()}, with a Board Readiness Score of "
        f"{risk['score']} out of 100."
    )

    if insight_block.get("strengths"):
        brief.append(
            f"Key positive signal includes "
            f"{insight_block['strengths'][0]['title'].lower()}."
        )

    if insight_block.get("risks"):
        brief.append(
            f"The primary risk relates to "
            f"{insight_block['risks'][0]['title'].lower()}, "
            "requiring timely leadership intervention."
        )

    brief.append(
        "Focused execution of the recommended actions over the next "
        "60–90 days can materially improve outcomes."
    )

    return " ".join(brief)


# =====================================================
# RECOMMENDATION NORMALIZATION (EXECUTIVE SAFE)
# =====================================================

def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    normalized: List[Dict[str, Any]] = []

    for r in recommendations[:5]:
        normalized.append({
            "priority": r.get("priority", "MEDIUM"),
            "action": r.get("action"),
            "owner": r.get("owner"),
            "timeline": r.get("timeline"),
            "goal": r.get("goal"),
            "confidence": round(float(r.get("confidence", 0.7)), 2),
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
