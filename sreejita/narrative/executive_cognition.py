# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (FINAL, GOVERNED)
# Sreejita Framework v3.5.x
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability


# =====================================================
# DOMAIN EXECUTIVE PROFILES (POLICY, NOT LOGIC)
# =====================================================

EXECUTIVE_DOMAIN_PROFILES = {
    "healthcare": {
        "escalate_info": True,
        "warning_penalty": 6,
        "risk_penalty": 12,
        "confidence_floor": 0.75,
        "mixed_domain_penalty": 1.0,
        "readiness_bias": -5,
        "tone": "clinical",
    },
    "retail": {
        "escalate_info": False,
        "warning_penalty": 3,
        "risk_penalty": 6,
        "confidence_floor": 0.65,
        "mixed_domain_penalty": 0.5,
        "readiness_bias": +10,
        "tone": "commercial",
    },
    "finance": {
        "escalate_info": False,
        "warning_penalty": 4,
        "risk_penalty": 8,
        "confidence_floor": 0.70,
        "mixed_domain_penalty": 0.6,
        "readiness_bias": +5,
        "tone": "financial",
    },
    "marketing": {
        "escalate_info": False,
        "warning_penalty": 2,
        "risk_penalty": 4,
        "confidence_floor": 0.60,
        "mixed_domain_penalty": 0.4,
        "readiness_bias": +15,
        "tone": "growth",
    },
    "supply_chain": {
        "escalate_info": True,
        "warning_penalty": 5,
        "risk_penalty": 10,
        "confidence_floor": 0.70,
        "mixed_domain_penalty": 0.8,
        "readiness_bias": 0,
        "tone": "operational",
    },
    "hr": {
        "escalate_info": False,
        "warning_penalty": 4,
        "risk_penalty": 7,
        "confidence_floor": 0.65,
        "mixed_domain_penalty": 0.5,
        "readiness_bias": +5,
        "tone": "people",
    },
    "customer": {
        "escalate_info": False,
        "warning_penalty": 3,
        "risk_penalty": 5,
        "confidence_floor": 0.60,
        "mixed_domain_penalty": 0.4,
        "readiness_bias": +10,
        "tone": "experience",
    },
}

DEFAULT_PROFILE = EXECUTIVE_DOMAIN_PROFILES["retail"]


def get_domain_profile(domain: str) -> Dict[str, Any]:
    return EXECUTIVE_DOMAIN_PROFILES.get(domain, DEFAULT_PROFILE)


# =====================================================
# EXECUTIVE RISK BANDS
# =====================================================

EXECUTIVE_RISK_BANDS = [
    (85, "LOW"),
    (70, "MEDIUM"),
    (50, "HIGH"),
    (0,  "CRITICAL"),
]


def derive_risk_level(score: int) -> Dict[str, Any]:
    score = int(score or 0)
    for threshold, label in EXECUTIVE_RISK_BANDS:
        if score >= threshold:
            return {"label": label, "score": score}
    return {"label": "CRITICAL", "score": score}


def infer_domain_from_kpis(kpis: Dict[str, Any]) -> str:
    return kpis.get("domain") or kpis.get("domain_name") or "retail"


# =====================================================
# KPI SELECTION
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap_map = kpis.get("_kpi_capabilities", {}) or {}
    conf_map = kpis.get("_confidence", {}) or {}

    ranked = []
    for key, capability in cap_map.items():
        value = kpis.get(key)
        if not isinstance(value, (int, float)):
            continue

        confidence = float(conf_map.get(key, 0.6))
        weight = {
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
            "rank_score": round(confidence * weight, 3),
        })

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    return ranked[:9]


# =====================================================
# INSIGHT STRUCTURING (FIXED — PHASE 2 VISIBLE)
# =====================================================

def structure_insights(
    insights: List[Dict[str, Any]],
    domain: str,
) -> Dict[str, Any]:

    profile = get_domain_profile(domain)
    insights = insights or []

    normalized = []

    for i in insights:
        if not isinstance(i, dict):
            continue

        lvl = i.get("level", "INFO")
        if lvl == "INFO" and profile["escalate_info"]:
            lvl = "WARNING"

        item = dict(i)
        item["level"] = lvl
        normalized.append(item)

    # -------------------------------
    # FORCE INCLUDE ONE COMPARATIVE INSIGHT
    # -------------------------------
    comparative_keywords = (
        "top vs long-tail",
        "category dominance",
        "variability",
    )

    comparative = [
        i for i in normalized
        if any(k in i.get("title", "").lower() for k in comparative_keywords)
    ][:1]

    # -------------------------------
    # STANDARD BUCKETING
    # -------------------------------
    strengths = comparative + [
        i for i in normalized
        if i["level"] in ("STRENGTH", "OPPORTUNITY")
        and i not in comparative
    ][:2]

    warnings = [i for i in normalized if i["level"] == "WARNING"][:2]
    risks = [i for i in normalized if i["level"] == "RISK"][:1]

    avg_conf = round(
        sum(float(i.get("confidence", 0.7)) for i in normalized)
        / max(len(normalized), 1),
        2,
    )

    return {
        "strengths": strengths,
        "warnings": warnings,
        "risks": risks,
        "composite": {
            "title": "Overall Executive Assessment",
            "summary": (
                "Performance shows measurable strengths with "
                "clear strategic leverage points."
            ),
            "confidence": avg_conf,
        },
    }

# =====================================================
# BOARD READINESS SCORE (DOMAIN-AWARE, HONEST)
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    domain: str,
) -> Dict[str, Any]:

    profile = get_domain_profile(domain)
    conf_map = kpis.get("_confidence", {}) or {}

    high_conf_kpis = [
        v for v in conf_map.values()
        if isinstance(v, (int, float)) and v >= profile["confidence_floor"]
    ]

    evidence_score = min(50, (len(high_conf_kpis) / 4) * 50)
    coverage_score = 25 if len(high_conf_kpis) >= 3 else 15

    warning_penalty = sum(
        profile["warning_penalty"]
        for i in insights if i.get("level") == "WARNING"
    )
    risk_penalty = sum(
        profile["risk_penalty"]
        for i in insights if i.get("level") == "RISK"
    )

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

    risk = derive_risk_level(score)

    return {
        "score": score,
        "band": risk["label"],
    }


# =====================================================
# EXECUTIVE BRIEF (CEO-LEGIBLE, DOMAIN-SAFE)
# =====================================================

def build_executive_brief(
    board_score: int,
    insight_block: Dict[str, Any],
    sub_domain: str,
    domain: str,
) -> str:

    profile = get_domain_profile(domain)
    risk = derive_risk_level(board_score)
    sub_domain = str(sub_domain).replace("_", " ")

    tone_prefix = {
        "clinical": "This assessment highlights key operational signals requiring leadership attention.",
        "commercial": "This performance review highlights current results and strategic growth opportunities.",
        "financial": "This financial overview summarizes current performance and optimization potential.",
        "growth": "This growth review highlights momentum and acceleration opportunities.",
        "operational": "This operational review outlines performance stability and efficiency drivers.",
        "people": "This workforce review highlights engagement and capability signals.",
        "experience": "This experience review summarizes customer engagement and loyalty indicators.",
    }.get(profile["tone"], "This performance review summarizes current outcomes.")

    brief: List[str] = [
        f"{tone_prefix} Overall readiness is assessed as "
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
        if not isinstance(r, dict):
            continue
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
# EXECUTIVE PAYLOAD (GLOBAL)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    domain: str = None,
) -> Dict[str, Any]:

    if not domain:
        domain = infer_domain_from_kpis(kpis)

    primary_sub = kpis.get("primary_sub_domain", "unknown")

    executive_kpis = select_executive_kpis(kpis)
    insight_block = structure_insights(insights or [], domain)
    board = compute_board_readiness_score(kpis, insights or [], domain)

    executive_brief = build_executive_brief(
        board_score=board["score"],
        insight_block=insight_block,
        sub_domain=primary_sub,
        domain=domain,
    )

    return {
        "executive_brief": executive_brief,
        "primary_kpis": executive_kpis,
        "insights": insight_block,
        "recommendations": normalize_recommendations(recommendations),
        "board_readiness": board,
        "sub_domain": primary_sub,
        "domain": domain,
    }

# =====================================================
# PER-SUB-DOMAIN EXECUTIVE COGNITION
# =====================================================

def build_subdomain_executive_payloads(
    *args,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Ultra backward-compatible executive payload builder.

    Accepts legacy and new call patterns, including:
      - (kpis, insights, recs)
      - (domain, kpis, insights, recs)
      - (kpis, insights, recs, config)
      - (domain, kpis, insights, recs, config)
    """

    domain = None
    kpis = None
    insights = []
    recommendations = []

    # ---- 1. Extract from positional args (best effort) ----
    for obj in args:
        if isinstance(obj, dict) and "sub_domains" in obj:
            kpis = obj
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            # Could be insights or recommendations
            if "level" in obj[0]:
                insights = obj
            elif "action" in obj[0]:
                recommendations = obj
        elif isinstance(obj, str):
            domain = obj

    # ---- 2. Keyword overrides ----
    domain = kwargs.get("domain") or domain
    kpis = kwargs.get("kpis") or kpis
    insights = kwargs.get("insights") or insights
    recommendations = kwargs.get("recommendations") or recommendations

    # ---- 3. Hard validation ----
    if not isinstance(kpis, dict):
        raise RuntimeError(
            "Executive cognition failed: unable to infer KPI payload "
            "from orchestrator call."
        )

    if not domain:
        domain = infer_domain_from_kpis(kpis)

    # ---- 4. Normal processing (unchanged logic) ----
    sub_domains = kpis.get("sub_domains", {}) or {}
    results: Dict[str, Dict[str, Any]] = {}

    domain_kpi_map = kpis.get("_domain_kpi_map", {}) or {}

    for sub in sub_domains.keys():

        sub_insights = [
            i for i in insights
            if isinstance(i, dict) and i.get("sub_domain") == sub
        ]

        sub_recs = [
            r for r in recommendations
            if isinstance(r, dict) and r.get("sub_domain") == sub
        ]

        allowed_kpis = set(domain_kpi_map.get(sub, []))

        sub_kpis = {
            k: v for k, v in kpis.items()
            if (
                k.startswith("_")
                or k in ["primary_sub_domain", "sub_domains"]
                or k in allowed_kpis
            )
        }

        sub_kpis["primary_sub_domain"] = sub

        results[sub] = build_executive_payload(
            sub_kpis,
            sub_insights,
            sub_recs,
            domain=domain,
        )

    return results

