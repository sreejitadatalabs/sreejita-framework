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
    Converts a confidence score into an executive risk band.
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
# RECOMMENDATION RANKING (EXECUTIVE SAFE)
# =====================================================

def rank_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deterministic executive ranking.
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
# EXECUTIVE KPI SELECTION (KEY-PRESERVING)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects 3–5 KPIs by executive relevance.
    Preserves original KPI keys for confidence lookup.
    """

    if not isinstance(kpis, dict):
        return []

    preferred_order = [
        "board_confidence_score",
        "maturity_level",
        "total_volume",
        "total_patients",
        "total_cost",
        "avg_cost_per_patient",
        "avg_duration",
        "avg_los",
        "readmission_rate",
        "variance_score",
    ]

    selected: List[Dict[str, Any]] = []

    for key in preferred_order:
        if key in kpis and kpis[key] is not None:
            selected.append({
                "key": key,
                "name": key.replace("_", " ").title(),
                "value": kpis[key],
            })
        if len(selected) >= 5:
            break

    # Fallback: magnitude-based
    if len(selected) < 3:
        numeric = [
            (k, v) for k, v in kpis.items()
            if isinstance(v, (int, float))
        ]
        numeric.sort(key=lambda x: abs(x[1]), reverse=True)

        existing = {x["key"] for x in selected}
        for k, v in numeric:
            if len(selected) >= 5:
                break
            if k not in existing:
                selected.append({
                    "key": k,
                    "name": k.replace("_", " ").title(),
                    "value": v,
                })

    return selected[:5]


# =====================================================
# INSIGHT PRIORITIZATION
# =====================================================

def extract_top_problems(
    insights: List[Dict[str, Any]]
) -> List[str]:
    """
    Returns top 3 executive problems by severity.
    """

    severity_rank = {
        "CRITICAL": 0,
        "RISK": 1,
        "WARNING": 2,
        "INFO": 3,
    }

    ordered = sorted(
        insights or [],
        key=lambda x: severity_rank.get(x.get("level", "INFO"), 3),
    )

    return [
        i.get("title", "Unlabeled Issue")
        for i in ordered[:3]
    ]


# =====================================================
# DECISION SNAPSHOT
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


# =====================================================
# SUCCESS CRITERIA
# =====================================================

def build_success_criteria(
    kpis: Dict[str, Any]
) -> List[str]:

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
    avg_conf = sum(conf.values()) / len(conf) if conf else 0.6
    kpi_score = avg_conf * 40

    # Insight trust (30)
    severity_weight = {
        "CRITICAL": 1.0,
        "RISK": 0.8,
        "WARNING": 0.6,
        "INFO": 0.4,
    }

    trusted, total = 0.0, 0.0
    for i in insights[:8]:
        w = severity_weight.get(i.get("level", "INFO"), 0.4)
        total += w
        if i.get("executive_summary_flag"):
            trusted += w

    insight_score = (trusted / total) * 30 if total else 15

    # Coverage (20)
    required = [
        "board_confidence_score",
        "total_volume",
        "avg_duration",
        "avg_unit_cost",
        "variance_score",
    ]
    present = sum(1 for k in required if k in kpis and kpis[k] is not None)
    coverage_score = (present / len(required)) * 20

    # Risk penalty
    criticals = sum(1 for i in insights if i.get("level") == "CRITICAL")
    penalty = 10 if criticals >= 3 else 6 if criticals == 2 else 3 if criticals == 1 else 0

    final = round(max(0, min(100, kpi_score + insight_score + coverage_score - penalty)))

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
# EXECUTIVE PAYLOAD (FINAL, CONFIDENCE-AWARE)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    confidence_map = kpis.get("_confidence", {})
    ranked_actions = rank_recommendations(recommendations)

    raw_kpis = select_executive_kpis(kpis)

    # Attach confidence to KPIs
    primary_kpis = []
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

    # Sort KPIs by impact × confidence
    primary_kpis.sort(
        key=lambda x: abs(float(x["value"])) * x["confidence"]
        if isinstance(x["value"], (int, float)) else 0,
        reverse=True
    )
    primary_kpis = primary_kpis[:5]

    snapshot = build_decision_snapshot(kpis, insights, recommendations)

    low_conf = [k["name"] for k in primary_kpis if k["confidence"] < 0.7]
    if low_conf:
        snapshot["confidence_note"] = (
            "Some indicators are based on limited or unstable data: "
            + ", ".join(low_conf[:3])
        )

    board_readiness = compute_board_readiness_score(kpis, insights)

    return {
        "snapshot": snapshot,
        "primary_kpis": primary_kpis,
        "top_problems": extract_top_problems(insights),
        "top_actions": [
            r.get("action", "Action required")
            for r in ranked_actions[:3]
        ],
        "success_criteria": build_success_criteria(kpis),
        "board_readiness": board_readiness,
        "_executive": {
            "primary_kpis": primary_kpis,
            "sub_domain": kpis.get("sub_domain"),
        },
    }
