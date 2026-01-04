# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (FINAL, GOVERNED)
# Sreejita Framework v3.5+
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability


# =====================================================
# EXECUTIVE RISK BANDS (BOARD LEGIBLE)
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
# EXECUTIVE KPI SELECTION (CAPABILITY-AWARE, MAX 9)
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

        ranked.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "value": round(value, 2),
            "capability": capability,
            "confidence": round(confidence, 2),
            "rank_score": round(confidence * capability_weight, 3),
        })

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)

    # 🔒 HARD RULE: EXECUTIVE MAX = 9 KPIs
    return ranked[:9]


# =====================================================
# INSIGHT STRUCTURING (EXECUTIVE FLOW)
# =====================================================

def structure_insights(insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    insights = insights or []

    strengths = [i for i in insights if i.get("level") == "STRENGTH"][:2]
    warnings  = [i for i in insights if i.get("level") == "WARNING"][:2]
    risks     = [i for i in insights if i.get("level") == "RISK"][:1]

    avg_conf = round(
        sum(float(i.get("confidence", 0.7)) for i in insights) / max(len(insights), 1),
        2,
    )

    return {
        "strengths": strengths,
        "warnings": warnings,
        "risks": risks,
        "composite": {
            "title": "Overall Executive Assessment",
            "summary": (
                "Operational performance shows measurable strengths, "
                "with identifiable risks requiring leadership attention."
            ),
            "confidence": avg_conf,
        },
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

    # Risk penalty (−10 per RISK insight)
    risk_penalty = sum(
        10 for i in insights if i.get("level") == "RISK"
    )

    score = round(
        max(0, min(100, evidence_score + coverage_score + 30 - risk_penalty))
    )

    # Governance cap for weak data
    if isinstance(kpis.get("data_completeness"), (int, float)):
        if kpis["data_completeness"] < 0.7:
            score = min(score, 60)

    band = (
        "BOARD READY" if score >= 85 else
        "REVIEW WITH CAUTION" if score >= 70 else
        "MANAGEMENT ONLY" if score >= 50 else
        "NOT BOARD SAFE"
    )

    return {
        "score": score,
        "band": band,
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
    sub_domain = sub_domain.replace("_", " ")

    brief: List[str] = [
        f"This {sub_domain} performance review is assessed as "
        f"{risk['label'].lower()}, with a Board Readiness Score of "
        f"{risk['score']} out of 100."
    ]

    if insight_block.get("strengths"):
        brief.append(
            f"A key strength observed is "
            f"{insight_block['strengths'][0]['title'].lower()}."
        )

    if insight_block.get("risks"):
        brief.append(
            f"The primary risk relates to "
            f"{insight_block['risks'][0]['title'].lower()}, "
            "requiring timely leadership attention."
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

    for r in (recommendations or [])[:5]:
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
# EXECUTIVE PAYLOAD (GLOBAL / SUB-DOMAIN)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    primary_sub = kpis.get("primary_sub_domain", "unknown")

    executive_kpis = select_executive_kpis(kpis)
    insight_block = structure_insights(insights or [])
    board = compute_board_readiness_score(kpis, insights or [])

    executive_brief = build_executive_brief(
        board_score=board["score"],
        insight_block=insight_block,
        sub_domain=primary_sub,
    )

    return {
        "executive_brief": executive_brief,
        "primary_kpis": executive_kpis,
        "insights": insight_block,
        "recommendations": normalize_recommendations(recommendations),
        "board_readiness": board,
        "sub_domain": primary_sub,
    }


# =====================================================
# PER-SUB-DOMAIN EXECUTIVE COGNITION (CRITICAL)
# =====================================================

def build_subdomain_executive_payloads(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:

    sub_domains = kpis.get("sub_domains", {}) or {}
    results: Dict[str, Dict[str, Any]] = {}

    for sub in sub_domains.keys():

        sub_insights = [
            i for i in (insights or [])
            if isinstance(i, dict) and i.get("sub_domain") == sub
        ]

        sub_recs = [
            r for r in (recommendations or [])
            if isinstance(r, dict) and r.get("sub_domain") == sub
        ]

        sub_kpis = dict(kpis)
        sub_kpis["primary_sub_domain"] = sub

        results[sub] = build_executive_payload(
            kpis=sub_kpis,
            insights=sub_insights,
            recommendations=sub_recs,
        )

    return results
