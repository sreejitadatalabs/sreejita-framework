from typing import Dict, Any, List

EXECUTIVE_RISK_BANDS = [
    (85, "LOW", "🟢"),
    (70, "MEDIUM", "🟡"),
    (50, "HIGH", "🟠"),
    (0,  "CRITICAL", "🔴"),
]

def derive_risk_level(score: int) -> Dict[str, str]:
    for threshold, label, icon in EXECUTIVE_RISK_BANDS:
        if score >= threshold:
            return {
                "label": label,
                "icon": icon,
                "score": score
            }

def build_decision_snapshot(kpis: Dict[str, Any],
                            insights: List[Dict],
                            recommendations: List[Dict]) -> Dict[str, Any]:

    score = kpis.get("board_confidence_score", 0)
    risk = derive_risk_level(score)

    critical_insights = [
        i for i in insights
        if i.get("level") in ("CRITICAL", "RISK")
    ][:3]

    top_actions = rank_recommendations(recommendations)[:3]

    return {
        "overall_risk": risk,
        "top_problems": [
            i["title"] for i in critical_insights
        ],
        "top_actions": [
            r["action"] for r in top_actions
        ],
        "decisions_required": [
            "Approve corrective initiative",
            "Assign executive owner",
            "Approve required resources"
        ]
    }

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    priority_order = [
        "board_confidence_score",
        "primary_efficiency_metric",
        "primary_cost_metric",
        "primary_volume_metric",
        "primary_quality_metric",
    ]

    selected = []
    for key in priority_order:
        if key in kpis:
            selected.append({
                "name": key.replace("_", " ").title(),
                "value": kpis[key]
            })

    # Fallback: pick numeric KPIs by impact
    if len(selected) < 3:
        numeric = [
            (k, v) for k, v in kpis.items()
            if isinstance(v, (int, float))
        ]
        numeric = sorted(numeric, key=lambda x: abs(x[1]), reverse=True)
        for k, v in numeric:
            if len(selected) >= 5:
                break
            selected.append({"name": k, "value": v})

    return selected[:5]

def extract_top_problems(insights: List[Dict]) -> List[str]:
    ordered = sorted(
        insights,
        key=lambda x: (
            x.get("level") != "CRITICAL",
            x.get("level") != "RISK"
        )
    )

    return [i["title"] for i in ordered[:3]]

def rank_recommendations(recommendations: List[Dict]) -> List[Dict]:
    def score(r):
        confidence = r.get("confidence", 0.5)
        priority_weight = {
            "CRITICAL": 3,
            "HIGH": 2,
            "MEDIUM": 1,
            "LOW": 0.5
        }.get(r.get("priority"), 1)

        return confidence * priority_weight

    return sorted(recommendations, key=score, reverse=True)

def build_success_criteria(kpis: Dict[str, Any]) -> List[str]:
    criteria = []

    if "board_confidence_score" in kpis:
        criteria.append(
            f"Confidence Score: {kpis['board_confidence_score']} → >65"
        )

    for k, v in kpis.items():
        if len(criteria) >= 4:
            break
        if isinstance(v, (int, float)) and v > 0:
            criteria.append(f"{k}: current → target")

    return criteria

def build_executive_payload(kpis, insights, recommendations) -> Dict[str, Any]:
    return {
        "decision_snapshot": build_decision_snapshot(
            kpis, insights, recommendations
        ),
        "executive_kpis": select_executive_kpis(kpis),
        "top_problems": extract_top_problems(insights),
        "top_actions": [
            r["action"] for r in rank_recommendations(recommendations)[:3]
        ],
        "success_criteria": build_success_criteria(kpis)
    }
