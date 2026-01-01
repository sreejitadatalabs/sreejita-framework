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
# EXECUTIVE KPI SELECTION (SUB-DOMAIN AWARE, SOFT)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects 3–5 KPIs by executive relevance.

    Strategy:
    - Confidence & maturity first
    - Scale & cost next
    - Duration / quality last
    - Fallback to magnitude
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

    # 1️⃣ Preferred KPIs
    for key in preferred_order:
        if key in kpis and kpis[key] is not None:
            selected.append({
                "name": key.replace("_", " ").title(),
                "value": kpis[key],
            })

        if len(selected) >= 5:
            break

    # 2️⃣ Fallback: highest-magnitude numeric KPIs
    if len(selected) < 3:
        numeric = [
            (k, v)
            for k, v in kpis.items()
            if isinstance(v, (int, float)) and v is not None
        ]
        numeric.sort(key=lambda x: abs(x[1]), reverse=True)

        existing = {
            x["name"].lower().replace(" ", "_")
            for x in selected
        }

        for k, v in numeric:
            if len(selected) >= 5:
                break
            if k not in existing:
                selected.append({
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
# EXECUTIVE KPI SELECTION (SUB-DOMAIN AWARE)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects 3–5 KPIs by executive relevance.
    """

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
                "name": key.replace("_", " ").title(),
                "key": key,
                "value": kpis[key],
            })
        if len(selected) >= 5:
            break

    return selected


# =====================================================
# INSIGHT PRIORITIZATION
# =====================================================

def extract_top_problems(
    insights: List[Dict[str, Any]]
) -> List[str]:

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


def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes a single board-readiness score (0–100).

    Dimensions:
    1. KPI confidence (40%)
    2. Insight confidence & severity (30%)
    3. Coverage completeness (20%)
    4. Risk concentration penalty (10%)
    """

    # -------------------------------------------------
    # 1. KPI CONFIDENCE SCORE (40)
    # -------------------------------------------------
    kpi_conf = kpis.get("_confidence", {})
    if kpi_conf:
        avg_kpi_conf = sum(kpi_conf.values()) / len(kpi_conf)
    else:
        avg_kpi_conf = 0.6  # safe fallback

    kpi_score = avg_kpi_conf * 40

    # -------------------------------------------------
    # 2. INSIGHT TRUST SCORE (30)
    # -------------------------------------------------
    severity_weight = {
        "CRITICAL": 1.0,
        "RISK": 0.8,
        "WARNING": 0.6,
        "INFO": 0.4,
    }

    trusted_insights = 0
    total_weight = 0.0

    for i in insights[:8]:  # only top insights matter
        sev = i.get("level", "INFO")
        weight = severity_weight.get(sev, 0.4)
        total_weight += weight

        # Executive-flagged insights imply higher trust
        if i.get("executive_summary_flag"):
            trusted_insights += weight

    if total_weight > 0:
        insight_score = (trusted_insights / total_weight) * 30
    else:
        insight_score = 15  # neutral fallback

    # -------------------------------------------------
    # 3. COVERAGE COMPLETENESS (20)
    # -------------------------------------------------
    required_sections = [
        "board_confidence_score",
        "total_volume",
        "avg_duration",
        "avg_unit_cost",
        "variance_score",
    ]

    present = sum(1 for k in required_sections if k in kpis and kpis[k] is not None)
    coverage_ratio = present / len(required_sections)
    coverage_score = coverage_ratio * 20

    # -------------------------------------------------
    # 4. RISK CONCENTRATION PENALTY (−10)
    # -------------------------------------------------
    critical_count = sum(1 for i in insights if i.get("level") == "CRITICAL")

    if critical_count >= 3:
        risk_penalty = 10
    elif critical_count == 2:
        risk_penalty = 6
    elif critical_count == 1:
        risk_penalty = 3
    else:
        risk_penalty = 0

    # -------------------------------------------------
    # FINAL SCORE
    # -------------------------------------------------
    raw_score = kpi_score + insight_score + coverage_score - risk_penalty
    final_score = max(0, min(100, round(raw_score)))

    return {
        "score": final_score,
        "band": (
            "BOARD READY" if final_score >= 85 else
            "REVIEW WITH CAUTION" if final_score >= 70 else
            "MANAGEMENT ONLY" if final_score >= 50 else
            "NOT BOARD SAFE"
        ),
        "components": {
            "kpi_confidence": round(kpi_score, 1),
            "insight_trust": round(insight_score, 1),
            "coverage": round(coverage_score, 1),
            "risk_penalty": -risk_penalty,
        },
    }
# =====================================================
# EXECUTIVE PAYLOAD (CONFIDENCE-AWARE)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    FINAL executive cognition contract (CONFIDENCE AWARE).
    """

    confidence_map = kpis.get("_confidence", {})
    ranked_actions = rank_recommendations(recommendations)

    raw_kpis = select_executive_kpis(kpis)

    # ---------------------------------------------
    # Attach confidence to KPIs
    # ---------------------------------------------
    primary_kpis = []
    for k in raw_kpis:
        key = k.get("key")
        conf = confidence_map.get(key, 0.6)

        primary_kpis.append({
            "name": k["name"],
            "value": k["value"],
            "confidence": round(conf, 2),
            "confidence_label": (
                "High" if conf >= 0.85 else
                "Moderate" if conf >= 0.70 else
                "Low"
            )
        })

    # ---------------------------------------------
    # Sort by confidence × magnitude
    # ---------------------------------------------
    def kpi_score(x):
        try:
            return abs(float(x["value"])) * x["confidence"]
        except Exception:
            return 0.0

    primary_kpis = sorted(
        primary_kpis,
        key=kpi_score,
        reverse=True
    )[:5]

    # ---------------------------------------------
    # Decision snapshot + confidence disclaimer
    # ---------------------------------------------
    snapshot = build_decision_snapshot(
        kpis, insights, recommendations
    )

    low_conf = [
        k["name"] for k in primary_kpis
        if k["confidence"] < 0.7
    ]

    if low_conf:
        snapshot["confidence_note"] = (
            "Some indicators are based on limited or unstable data: "
            + ", ".join(low_conf[:3])
        )

    board_readiness = compute_board_readiness_score(
    kpis=kpis,
    insights=insights,
    )
    
    return {
        # 🔑 Snapshot
        "snapshot": build_decision_snapshot(
            kpis, insights, recommendations
        ),
    
        # 🔑 KPIs
        "primary_kpis": primary_kpis,
    
        # 🔑 Problems & Actions
        "top_problems": extract_top_problems(insights),
        "top_actions": [
            r.get("action", "Action required")
            for r in ranked_actions[:3]
        ],
    
        # 🔑 Outcome framing
        "success_criteria": build_success_criteria(kpis),
    
        # 🏛️ BOARD READINESS (NEW)
        "board_readiness": board_readiness,
    
        # 🔒 INTERNAL METADATA
        "_executive": {
            "primary_kpis": primary_kpis,
            "sub_domain": kpis.get("sub_domain"),
        },
    }
