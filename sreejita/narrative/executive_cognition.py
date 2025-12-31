from typing import Dict, Any, List

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
    """
    Converts a confidence score into an executive risk band.
    Guaranteed safe return.
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
                "score": score
            }

    # Absolute fallback (should never hit)
    return {"label": "CRITICAL", "icon": "🔴", "score": score}


# =====================================================
# RECOMMENDATION RANKING
# =====================================================

def rank_recommendations(recommendations: List[Dict]) -> List[Dict]:
    """
    Ranks recommendations by executive impact.
    Deterministic, confidence-weighted.
    """

    def score(r: Dict[str, Any]) -> float:
        confidence = float(r.get("confidence", 0.5))

        priority_weight = {
            "CRITICAL": 3.0,
            "HIGH": 2.0,
            "MEDIUM": 1.0,
            "LOW": 0.5
        }.get(r.get("priority", "MEDIUM"), 1.0)

        return confidence * priority_weight

    return sorted(recommendations or [], key=score, reverse=True)


# =====================================================
# EXECUTIVE KPI SELECTION
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects 3–5 KPIs suitable for executive consumption.
    Uses dynamic fallback if canonical KPIs are missing.
    """

    canonical_keys = [
        "board_confidence_score",
        "avg_los",
        "avg_cost_per_patient",
        "total_patients",
        "readmission_rate",
    ]

    selected = []

    for key in canonical_keys:
        if key in kpis and kpis[key] is not None:
            selected.append({
                "name": key.replace("_", " ").title(),
                "value": kpis[key]
            })

    # Fallback: highest-magnitude numeric KPIs
    if len(selected) < 3:
        numeric = [
            (k, v) for k, v in kpis.items()
            if isinstance(v, (int, float))
        ]
        numeric = sorted(numeric, key=lambda x: abs(x[1]), reverse=True)

        for k, v in numeric:
            if len(selected) >= 5:
                break
            if k not in {x["name"].lower().replace(" ", "_") for x in selected}:
                selected.append({
                    "name": k.replace("_", " ").title(),
                    "value": v
                })

    return selected[:5]


# =====================================================
# INSIGHT PRIORITIZATION
# =====================================================

def extract_top_problems(insights: List[Dict]) -> List[str]:
    """
    Extracts top executive problems in severity order.
    """

    severity_rank = {
        "CRITICAL": 0,
        "RISK": 1,
        "WARNING": 2,
        "INFO": 3
    }

    ordered = sorted(
        insights or [],
        key=lambda x: severity_rank.get(x.get("level", "INFO"), 3)
    )

    return [i.get("title", "Unlabeled Issue") for i in ordered[:3]]


# =====================================================
# DECISION SNAPSHOT
# =====================================================

def build_decision_snapshot(kpis, insights, recommendations):
    score = kpis.get("board_confidence_score", 0)
    risk = derive_risk_level(score)

    critical_insights = [
        i for i in insights
        if i.get("level") in ("CRITICAL", "RISK")
    ][:3]

    return {
        "title": "EXECUTIVE DECISION SNAPSHOT",
        "overall_risk": f"{risk['icon']} {risk['label']} ({score} / 100)",
        "top_problems": [i["title"] for i in critical_insights],
        "decisions_required": [
            "Approve LOS reduction program",
            "Assign executive owner",
            "Approve capacity optimization budget"
        ]
    }

# =====================================================
# SUCCESS CRITERIA
# =====================================================

def build_success_criteria(kpis: Dict[str, Any]) -> List[str]:
    """
    Converts KPIs into measurable executive success signals.
    """
    criteria = []

    if "board_confidence_score" in kpis:
        criteria.append("Board Confidence Score → >70")

    for k, v in kpis.items():
        if len(criteria) >= 4:
            break
        if isinstance(v, (int, float)) and v > 0:
            criteria.append(f"{k.replace('_', ' ').title()}: improve vs baseline")

    return criteria


# =====================================================
# EXECUTIVE PAYLOAD (FINAL OUTPUT)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict],
    recommendations: List[Dict]
) -> Dict[str, Any]:
    """
    Single executive-ready cognition object.
    """

    ranked_actions = rank_recommendations(recommendations)

    return {
        "decision_snapshot": build_decision_snapshot(
            kpis, insights, recommendations
        ),
        "executive_kpis": select_executive_kpis(kpis),
        "top_problems": extract_top_problems(insights),
        "top_actions": [r.get("action") for r in ranked_actions[:3]],
        "success_criteria": build_success_criteria(kpis)
    }
