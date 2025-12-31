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


def derive_risk_level(score: Any) -> Dict[str, Any]:
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
                "score": score,
            }

    return {"label": "CRITICAL", "icon": "🔴", "score": score}


# =====================================================
# RECOMMENDATION RANKING
# =====================================================

def rank_recommendations(recommendations: List[Dict]) -> List[Dict]:
    """
    Ranks recommendations by executive impact.
    Deterministic, confidence-weighted.
    """

    def _score(rec: Dict[str, Any]) -> float:
        try:
            confidence = float(rec.get("confidence", 0.5))
        except Exception:
            confidence = 0.5

        priority_weight = {
            "CRITICAL": 3.0,
            "HIGH": 2.0,
            "MEDIUM": 1.0,
            "LOW": 0.5,
        }.get(rec.get("priority", "MEDIUM"), 1.0)

        return confidence * priority_weight

    return sorted(recommendations or [], key=_score, reverse=True)


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

    selected: List[Dict[str, Any]] = []

    # 1️⃣ Canonical KPIs first
    for key in canonical_keys:
        if key in kpis and kpis[key] is not None:
            selected.append({
                "name": key.replace("_", " ").title(),
                "value": kpis[key],
            })

    # 2️⃣ Fallback: highest-impact numeric KPIs
    if len(selected) < 3:
        numeric = [
            (k, v) for k, v in kpis.items()
            if isinstance(v, (int, float)) and v is not None
        ]
        numeric.sort(key=lambda x: abs(x[1]), reverse=True)

        existing_keys = {x["name"].lower().replace(" ", "_") for x in selected}

        for k, v in numeric:
            if len(selected) >= 5:
                break
            if k not in existing_keys:
                selected.append({
                    "name": k.replace("_", " ").title(),
                    "value": v,
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
# DECISION SNAPSHOT (EXECUTIVE-FIRST)
# =====================================================

def build_decision_snapshot(
    kpis: Dict[str, Any],
    insights: List[Dict],
    recommendations: List[Dict],
) -> Dict[str, Any]:
    """
    Constructs a decisive, scan-ready executive snapshot.
    """

    score = kpis.get("board_confidence_score", 0)
    risk = derive_risk_level(score)

    top_problems = extract_top_problems(insights)

    return {
        "title": "EXECUTIVE DECISION SNAPSHOT",
        "overall_risk": f"{risk['icon']} {risk['label']} (Score: {risk['score']} / 100)",
        "top_problems": top_problems,
        "decisions_required": [
            "Approve corrective initiative",
            "Assign executive owner",
            "Approve required resources",
        ],
    }


# =====================================================
# SUCCESS CRITERIA
# =====================================================

def build_success_criteria(kpis: Dict[str, Any]) -> List[str]:
    """
    Converts KPIs into measurable executive success signals.
    """
    criteria: List[str] = []

    score = kpis.get("board_confidence_score")
    if isinstance(score, (int, float)):
        criteria.append("Board Confidence Score → >70")

    for k, v in kpis.items():
        if len(criteria) >= 4:
            break
        if isinstance(v, (int, float)) and v > 0:
            criteria.append(f"{k.replace('_', ' ').title()} → Improve vs baseline")

    return criteria


# =====================================================
# EXECUTIVE PAYLOAD (FINAL CONTRACT)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict],
    recommendations: List[Dict],
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
        "top_actions": [
            r.get("action", "Action required")
            for r in ranked_actions[:3]
        ],
        "success_criteria": build_success_criteria(kpis),
    }
