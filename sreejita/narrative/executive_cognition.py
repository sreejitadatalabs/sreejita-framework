from typing import Dict, Any, List


# =====================================================
# EXECUTIVE RISK MODEL (CANONICAL)
# =====================================================

EXECUTIVE_RISK_BANDS = [
    (85, "LOW", "🟢"),
    (70, "MEDIUM", "🟡"),
    (50, "HIGH", "🟠"),
    (0,  "CRITICAL", "🔴"),
]


def derive_risk_level(score: Any) -> Dict[str, Any]:
    """
    Converts a numeric score into an executive risk band.
    ALWAYS safe.
    """
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
# RECOMMENDATION RANKING
# =====================================================

def rank_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deterministic executive ranking based on Priority × Confidence.
    """

    PRIORITY_WEIGHT = {
        "CRITICAL": 3.0,
        "HIGH": 2.0,
        "MEDIUM": 1.0,
        "LOW": 0.5,
    }

    def score(rec: Dict[str, Any]) -> float:
        try:
            confidence = float(rec.get("confidence", 0.6))
        except Exception:
            confidence = 0.6

        priority = rec.get("priority", "MEDIUM")
        weight = PRIORITY_WEIGHT.get(priority, 1.0)

        return confidence * weight

    return sorted(recommendations or [], key=score, reverse=True)


# =====================================================
# EXECUTIVE KPI SELECTION
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects 3–5 KPIs based on capability priority, confidence, and relevance.
    """
    if not isinstance(kpis, dict):
        return []

    cap_map = kpis.get("_kpi_capabilities", {})
    conf_map = kpis.get("_confidence", {})
    scored = []

    for key, cap in cap_map.items():
        value = kpis.get(key)
        if value is None or not isinstance(value, (int, float)):
            continue

        confidence = conf_map.get(key, 0.6)
        # Weight by inverse of required confidence (lower req = higher base importance)
        priority = get_capability_spec(Capability(cap)).min_confidence
        score = abs(value) * confidence * (1 / priority)

        scored.append({
            "key": key,
            "name": key.replace("_", " ").title(),
            "value": value,
            "confidence": round(confidence, 2),
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]

# =====================================================
# INSIGHT PRIORITIZATION & CONFIDENCE WEIGHTING
# =====================================================

def confidence_weight_insights(
    insights: List[Dict[str, Any]],
    kpi_confidence: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Re-ranks insights so low-confidence alerts do not dominate.
    Idempotent & safe.
    """

    SEVERITY_WEIGHT = {
        "CRITICAL": 4,
        "RISK": 3,
        "WARNING": 2,
        "INFO": 1,
    }

    def infer_confidence(insight: Dict[str, Any]) -> float:
        title = insight.get("title", "").lower()

        if "cost" in title:
            return kpi_confidence.get("avg_unit_cost", 0.6)
        if "flow" in title or "duration" in title or "stay" in title:
            return kpi_confidence.get("avg_duration", 0.6)
        if "quality" in title or "readmission" in title:
            return kpi_confidence.get("adverse_event_rate", 0.6)
        if "variance" in title:
            return kpi_confidence.get("variance_score", 0.6)
        if "data" in title:
            return kpi_confidence.get("data_completeness", 0.7)

        return 0.65

    def score(i: Dict[str, Any]) -> float:
        severity = SEVERITY_WEIGHT.get(i.get("level", "INFO"), 1)
        conf = infer_confidence(i)
        boost = 1.2 if i.get("executive_summary_flag") else 1.0
        return severity * conf * boost

    return sorted(insights or [], key=score, reverse=True)


def extract_top_problems(insights: List[Dict[str, Any]]) -> List[str]:
    """
    Returns top 3 problems by severity ordering.
    """
    return [
        i.get("title", "Unlabeled Issue")
        for i in (insights or [])[:3]
    ]


# =====================================================
# SNAPSHOT BUILDERS
# =====================================================

def build_decision_snapshot(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    score = kpis.get("board_confidence_score", 0)
    risk = derive_risk_level(score)
    ranked_actions = rank_recommendations(recommendations)

    return {
        "title": "EXECUTIVE DECISION SNAPSHOT",
        "overall_risk": risk["display"],
        "top_problems": extract_top_problems(insights),
        "top_actions": [
            r.get("action", "Action required")
            for r in ranked_actions[:3]
        ],
        "decisions_required": [
            "Approve corrective initiative",
            "Assign executive owner",
            "Approve required resources",
        ],
    }


def build_success_criteria(kpis: Dict[str, Any]) -> List[str]:
    criteria: List[str] = []

    score = kpis.get("board_confidence_score")
    if isinstance(score, (int, float)):
        criteria.append("Board Confidence Score → >70")

    for k, v in kpis.items():
        if len(criteria) >= 4:
            break
        if isinstance(v, (int, float)) and v > 0:
            criteria.append(
                f"{k.replace('_', ' ').title()} → Improve vs baseline"
            )

    return criteria


# =====================================================
# BOARD READINESS SCORE
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:

    # KPI confidence (40)
    conf = kpis.get("_confidence", {})
    avg_conf = (
        sum(conf.values()) / len(conf)
        if conf else 0.6
    )
    kpi_score = avg_conf * 40

    # Insight trust (30)
    total = max(len(insights[:8]), 1)
    trusted = sum(
        1 for i in insights[:8]
        if i.get("executive_summary_flag")
    )
    insight_score = (trusted / total) * 30

    # Coverage (30)
    required = [
        "board_confidence_score",
        "total_volume",
        "avg_duration",
        "avg_unit_cost",
        "variance_score",
    ]
    present = sum(
        1 for k in required
        if k in kpis and kpis[k] is not None
    )
    coverage_score = (present / len(required)) * 30

    # Penalty
    criticals = sum(
        1 for i in insights
        if i.get("level") == "CRITICAL"
    )
    penalty = 10 if criticals >= 3 else 5 if criticals > 0 else 0

    final = round(
        max(0, min(100, kpi_score + insight_score + coverage_score - penalty))
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
        "components": {
            "kpi": round(kpi_score, 1),
            "insight": round(insight_score, 1),
            "coverage": round(coverage_score, 1),
            "penalty": -penalty,
        },
    }


# =====================================================
# EXECUTIVE PAYLOAD (SINGLE SOURCE OF TRUTH)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    confidence_map = kpis.get("_confidence", {})

    # 1. Insight ordering
    insights = confidence_weight_insights(insights, confidence_map)

    # 2. KPI selection + confidence
    raw_kpis = select_executive_kpis(kpis)

    primary_kpis: List[Dict[str, Any]] = []
    for k in raw_kpis:
        key = k["key"]
        conf = confidence_map.get(key, 0.6)

        primary_kpis.append({
            "name": k["name"],
            "value": k["value"],
            "confidence": round(conf, 2),
            "confidence_label": (
                "High" if conf >= 0.85 else
                "Moderate" if conf >= 0.70 else
                "Low"
            ),
        })

    primary_kpis.sort(
        key=lambda x: (
            abs(float(x["value"])) * x["confidence"]
            if isinstance(x["value"], (int, float)) else 0
        ),
        reverse=True,
    )
    primary_kpis = primary_kpis[:5]

    snapshot = build_decision_snapshot(kpis, insights, recommendations)

    low_conf = [
        k["name"] for k in primary_kpis
        if k["confidence"] < 0.7
    ]
    if low_conf:
        snapshot["confidence_note"] = (
            "⚠️ Data Stability Warning: "
            + ", ".join(low_conf[:3])
        )

    return {
        "snapshot": snapshot,
        "primary_kpis": primary_kpis,
        "top_problems": extract_top_problems(insights),
        "top_actions": [
            r.get("action", "Action required")
            for r in rank_recommendations(recommendations)[:3]
        ],
        "success_criteria": build_success_criteria(kpis),
        "board_readiness": compute_board_readiness_score(kpis, insights),
        "_executive": {
            "primary_kpis": primary_kpis,
            "sub_domain": kpis.get("primary_sub_domain"),
        },
    }
