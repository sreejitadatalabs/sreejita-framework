from typing import Dict, Any, List
from sreejita.core.capabilities import (
    Capability,
    get_capability_spec,
)

# =====================================================
# EXECUTIVE RISK MODEL
# =====================================================

EXECUTIVE_RISK_BANDS = [
    (85, "LOW", "🟢"),
    (70, "MEDIUM", "🟡"),
    (50, "HIGH", "🟠"),
    (0,  "CRITICAL", "🔴"),
]


def derive_risk_level(score: Any) -> Dict[str, Any]:
    try:
        score = int(score)
    except Exception:
        score = 0

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
# KPI SELECTION (MAX 9, EXECUTIVE SAFE)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(kpis, dict) or "_kpi_capabilities" not in kpis:
        return []

    cap_map = kpis.get("_kpi_capabilities", {})
    conf_map = kpis.get("_confidence", {})
    scored = []

    for key, cap in cap_map.items():
        value = kpis.get(key)
        if not isinstance(value, (int, float)):
            continue

        confidence = conf_map.get(key, 0.6)
        cap_spec = get_capability_spec(Capability(cap))

        score = abs(value) * confidence * (1 / cap_spec.min_confidence)

        scored.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "value": value,
            "confidence": round(confidence, 2),
            "capability": cap,
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:9]   # 🔒 HARD MAX 9


# =====================================================
# INSIGHT STRUCTURING (EXECUTIVE FRIENDLY)
# =====================================================

def structure_insights(
    insights: List[Dict[str, Any]],
    kpi_confidence: Dict[str, float],
) -> Dict[str, List[Dict[str, Any]]]:

    strengths = []
    risks = []

    for ins in insights:
        level = ins.get("level", "").upper()

        if level in ("INFO", "STABLE"):
            strengths.append(ins)
        else:
            risks.append(ins)

    strengths = strengths[:2]   # Max 2 positives
    risks = risks[:3]           # Max 3 risks

    composite = {
        "title": "Composite Executive Insight",
        "summary": (
            "Overall performance shows identifiable strengths, "
            "with emerging risks that are manageable through "
            "targeted intervention."
        ),
        "confidence": round(
            sum(kpi_confidence.values()) / max(len(kpi_confidence), 1), 2
        ),
    }

    return {
        "strengths": strengths,
        "risks": risks,
        "composite": composite,
    }


# =====================================================
# 1-MINUTE EXECUTIVE BRIEF
# =====================================================

def build_executive_brief(
    kpis: Dict[str, Any],
    insight_block: Dict[str, Any],
) -> str:
    score = kpis.get("board_confidence_score", 0)
    risk = derive_risk_level(score)

    positives = insight_block["strengths"]
    risks = insight_block["risks"]

    brief = [
        f"Current operational performance is assessed as {risk['label'].lower()}, "
        f"with a Board Readiness Score of {risk['score']} / 100."
    ]

    if positives:
        brief.append(
            f"Key strengths include {positives[0]['title'].lower()}."
        )

    if risks:
        brief.append(
            f"Primary risk relates to {risks[0]['title'].lower()}, "
            "which requires timely management attention."
        )

    brief.append(
        "Overall, the situation is controllable with focused execution "
        "over the next planning cycle."
    )

    return " ".join(brief)


# =====================================================
# RECOMMENDATION NORMALIZATION (EXECUTIVE ACTION)
# =====================================================

def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    normalized = []

    for r in recommendations[:5]:
        normalized.append({
            "priority": r.get("priority", "MEDIUM"),
            "action": r.get("action", "Action required"),
            "owner": r.get("owner", "Operations"),
            "timeline": r.get("timeline", "Next 30–60 days"),
            "goal": r.get(
                "goal",
                "Improve stability and reduce operational risk",
            ),
            "confidence": round(r.get("confidence", 0.6), 2),
        })

    return normalized


# =====================================================
# BOARD READINESS SCORE
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    conf = kpis.get("_confidence", {})
    avg_conf = sum(conf.values()) / max(len(conf), 1)
    kpi_score = avg_conf * 40

    coverage_keys = [
        "total_volume",
        "avg_duration",
        "avg_unit_cost",
        "variance_score",
    ]
    present = sum(1 for k in coverage_keys if k in kpis)
    coverage_score = (present / len(coverage_keys)) * 30

    criticals = sum(1 for i in insights if i.get("level") == "CRITICAL")
    penalty = 10 if criticals else 0

    final = round(
        max(0, min(100, kpi_score + coverage_score + 30 - penalty))
    )

    if kpis.get("data_completeness", 0) < 0.7:
        final = min(final, 60)

    return {
        "score": final,
        "band": (
            "BOARD READY" if final >= 85 else
            "REVIEW WITH CAUTION" if final >= 70 else
            "MANAGEMENT ONLY" if final >= 50 else
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

    confidence_map = kpis.get("_confidence", {})

    # 1. KPI SELECTION
    primary_kpis = select_executive_kpis(kpis)

    # 2. INSIGHT STRUCTURE
    insight_block = structure_insights(insights, confidence_map)

    # 3. EXECUTIVE BRIEF
    executive_brief = build_executive_brief(kpis, insight_block)

    # 4. RECOMMENDATIONS
    executive_recs = normalize_recommendations(recommendations)

    # 5. BOARD READINESS
    board = compute_board_readiness_score(
        kpis,
        insights,
    )

    return {
        "executive_brief": executive_brief,
        "primary_kpis": primary_kpis,
        "insights": insight_block,
        "recommendations": executive_recs,
        "board_readiness": board,
        "sub_domain": kpis.get("primary_sub_domain"),
    }
