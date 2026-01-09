from typing import Dict, Any, List

# =====================================================
# DOMAIN EXECUTIVE PROFILES (POLICY ONLY)
# =====================================================

EXECUTIVE_PROFILES = {
    "healthcare": {
        "risk_tolerance": "low",
        "escalate_info": True,
        "mixed_domain_penalty": 1.0,
        "min_kpis_for_confidence": 5,
        "confidence_floor": 0.75,
        "readiness_bias": -5,
        "tone": "clinical",
    },
    "retail": {
        "risk_tolerance": "medium",
        "escalate_info": False,
        "mixed_domain_penalty": 0.5,
        "min_kpis_for_confidence": 3,
        "confidence_floor": 0.65,
        "readiness_bias": +10,
        "tone": "commercial",
    },
    "finance": {
        "risk_tolerance": "medium",
        "escalate_info": False,
        "mixed_domain_penalty": 0.6,
        "min_kpis_for_confidence": 4,
        "confidence_floor": 0.7,
        "readiness_bias": +5,
        "tone": "financial",
    },
    "marketing": {
        "risk_tolerance": "high",
        "escalate_info": False,
        "mixed_domain_penalty": 0.4,
        "min_kpis_for_confidence": 3,
        "confidence_floor": 0.6,
        "readiness_bias": +15,
        "tone": "growth",
    },
    "supply_chain": {
        "risk_tolerance": "medium",
        "escalate_info": True,
        "mixed_domain_penalty": 0.8,
        "min_kpis_for_confidence": 4,
        "confidence_floor": 0.7,
        "readiness_bias": 0,
        "tone": "operational",
    },
    "hr": {
        "risk_tolerance": "medium",
        "escalate_info": False,
        "mixed_domain_penalty": 0.5,
        "min_kpis_for_confidence": 3,
        "confidence_floor": 0.65,
        "readiness_bias": +5,
        "tone": "people",
    },
    "customer": {
        "risk_tolerance": "high",
        "escalate_info": False,
        "mixed_domain_penalty": 0.4,
        "min_kpis_for_confidence": 3,
        "confidence_floor": 0.6,
        "readiness_bias": +10,
        "tone": "experience",
    },
}

DEFAULT_PROFILE = EXECUTIVE_PROFILES["retail"]


def get_domain_profile(domain: str) -> Dict[str, Any]:
    return EXECUTIVE_PROFILES.get(domain, DEFAULT_PROFILE)


# =====================================================
# RISK BANDING
# =====================================================

def derive_risk_level(score: int) -> Dict[str, str]:
    if score >= 75:
        return {"label": "HIGH", "color": "green"}
    if score >= 55:
        return {"label": "MEDIUM", "color": "orange"}
    return {"label": "LOW", "color": "red"}


# =====================================================
# BOARD READINESS SCORE (UNIVERSAL)
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    domain: str,
) -> Dict[str, Any]:

    profile = get_domain_profile(domain)
    conf_map = kpis.get("_confidence", {}) or {}

    strong_kpis = [
        v for v in conf_map.values()
        if isinstance(v, (int, float)) and v >= profile["confidence_floor"]
    ]

    evidence_score = min(50, (len(strong_kpis) / profile["min_kpis_for_confidence"]) * 50)
    coverage_score = 25 if len(strong_kpis) >= profile["min_kpis_for_confidence"] else 15

    warning_penalty = sum(3 for i in insights if i.get("level") == "WARNING")
    risk_penalty = sum(6 for i in insights if i.get("level") == "RISK")

    score = round(
        max(
            0,
            min(
                100,
                evidence_score
                + coverage_score
                + profile["readiness_bias"]
                - warning_penalty
                - risk_penalty,
            ),
        )
    )

    if kpis.get("primary_sub_domain") == "mixed":
        score = round(score - (10 * profile["mixed_domain_penalty"]))

    if isinstance(kpis.get("data_completeness"), (int, float)):
        if kpis["data_completeness"] < 0.6:
            score = min(score, 65)

    band = derive_risk_level(score)

    return {
        "score": score,
        "band": band["label"],
        "color": band["color"],
    }


# =====================================================
# INSIGHT STRUCTURING (NON-BIASED)
# =====================================================

def structure_insights(
    insights: List[Dict[str, Any]],
    domain: str,
) -> Dict[str, Any]:

    profile = get_domain_profile(domain)
    insights = insights or []

    normalized: List[Dict[str, Any]] = []

    for i in insights:
        if not isinstance(i, dict):
            continue

        lvl = i.get("level", "INFO")

        if lvl == "INFO" and profile["escalate_info"]:
            lvl = "WARNING"

        item = dict(i)
        item["level"] = lvl
        normalized.append(item)

    strengths = [i for i in normalized if i["level"] == "STRENGTH"][:2]
    warnings = [i for i in normalized if i["level"] == "WARNING"][:2]
    risks = [i for i in normalized if i["level"] == "RISK"][:1]

    avg_conf = round(
        sum(float(i.get("confidence", 0.75)) for i in normalized)
        / max(len(normalized), 1),
        2,
    )

    tone = profile["tone"]

    summary_map = {
        "clinical": "Operational signals indicate areas requiring close monitoring and structured intervention.",
        "commercial": "Performance shows measurable strengths with clear opportunities for growth.",
        "financial": "Financial indicators reflect a stable position with targeted optimization potential.",
        "growth": "Growth momentum is visible, with opportunities to accelerate impact.",
        "operational": "Operational performance is generally stable with identifiable efficiency levers.",
        "people": "People metrics suggest balanced workforce dynamics with improvement opportunities.",
        "experience": "Customer experience signals show engagement strength with areas to enhance loyalty.",
    }

    return {
        "strengths": strengths,
        "warnings": warnings,
        "risks": risks,
        "composite": {
            "title": "Overall Executive Assessment",
            "summary": summary_map.get(tone, summary_map["commercial"]),
            "confidence": avg_conf,
        },
    }


# =====================================================
# EXECUTIVE BRIEF (CLIENT-SAFE)
# =====================================================

def build_executive_brief(
    domain: str,
    readiness: Dict[str, Any],
    kpis: Dict[str, Any],
) -> str:

    tone = get_domain_profile(domain)["tone"]
    band = readiness["band"]

    phrasing = {
        "clinical": "This assessment highlights key operational signals requiring leadership attention.",
        "commercial": "This performance review highlights current results and strategic growth opportunities.",
        "financial": "This financial overview summarizes current performance and optimization potential.",
        "growth": "This growth review highlights momentum and acceleration opportunities.",
        "operational": "This operational review outlines performance stability and efficiency drivers.",
        "people": "This workforce review highlights engagement and capability signals.",
        "experience": "This experience review summarizes customer engagement and loyalty indicators.",
    }

    return (
        f"{phrasing.get(tone, phrasing['commercial'])} "
        f"Overall readiness is assessed as {band}."
    )


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def run_executive_cognition(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    structured = structure_insights(insights, domain)
    readiness = compute_board_readiness_score(kpis, insights, domain)
    brief = build_executive_brief(domain, readiness, kpis)

    return {
        "executive_brief": brief,
        "board_readiness": readiness,
        "insights": structured,
    }
