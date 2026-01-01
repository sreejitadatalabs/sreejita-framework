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

def rank_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic executive ranking based on Priority * Confidence.
    """
    PRIORITY_WEIGHT = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}

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
    Selects 3–5 KPIs by executive relevance.
    """
    preferred_order = [
        "board_confidence_score", "maturity_level", "total_volume", 
        "total_patients", "total_cost", "avg_cost_per_patient", 
        "avg_duration", "avg_los", "readmission_rate", "variance_score",
    ]

    selected: List[Dict[str, Any]] = []

    for key in preferred_order:
        if key in kpis and kpis[key] is not None:
            selected.append({
                "name": key.replace("_", " ").title(),
                "key": key,
                "value": kpis[key],
            })
        if len(selected) >= 5: break

    # Fallback to magnitude if needed
    if len(selected) < 3:
        numeric = [(k, v) for k, v in kpis.items() if isinstance(v, (int, float))]
        numeric.sort(key=lambda x: abs(x[1]), reverse=True)
        existing = {x["key"] for x in selected}
        
        for k, v in numeric:
            if len(selected) >= 5: break
            if k not in existing:
                selected.append({
                    "name": k.replace("_", " ").title(),
                    "key": k,
                    "value": v
                })

    return selected[:5]

# =====================================================
# INSIGHT PRIORITIZATION & CONFIDENCE WEIGHTING
# =====================================================

def confidence_weight_insights(
    insights: List[Dict[str, Any]],
    kpi_confidence: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Re-ranks insights so low-confidence alerts don't scream "CRITICAL".
    """
    SEVERITY_WEIGHT = {"CRITICAL": 4, "RISK": 3, "WARNING": 2, "INFO": 1}

    def infer_confidence(insight):
        title = insight.get("title", "").lower()
        if "cost" in title: return kpi_confidence.get("avg_unit_cost", 0.6)
        if "flow" in title: return kpi_confidence.get("avg_duration", 0.6)
        if "data" in title: return kpi_confidence.get("data_completeness", 0.7)
        return 0.65

    def score(i):
        severity = SEVERITY_WEIGHT.get(i.get("level", "INFO"), 1)
        conf = infer_confidence(i)
        boost = 1.2 if i.get("executive_summary_flag") else 1.0
        return severity * conf * boost

    return sorted(insights, key=score, reverse=True)

def extract_top_problems(insights: List[Dict[str, Any]]) -> List[str]:
    return [i.get("title", "Unlabeled Issue") for i in insights[:3]]

# =====================================================
# SNAPSHOTS & SCORING
# =====================================================

def build_decision_snapshot(kpis, insights, recommendations):
    score = kpis.get("board_confidence_score", 0)
    risk = derive_risk_level(score)
    ranked_actions = rank_recommendations(recommendations)

    return {
        "title": "EXECUTIVE DECISION SNAPSHOT",
        "overall_risk": risk["display"],
        "top_problems": extract_top_problems(insights),
        "top_actions": [r.get("action") for r in ranked_actions[:3]],
        "decisions_required": ["Approve corrective initiative", "Assign executive owner", "Approve resources"],
    }

def build_success_criteria(kpis):
    criteria = []
    if kpis.get("board_confidence_score"): criteria.append("Board Confidence Score → >70")
    for k, v in kpis.items():
        if len(criteria) >= 4: break
        if isinstance(v, (int, float)) and v > 0:
            criteria.append(f"{k.replace('_', ' ').title()} → Improve vs baseline")
    return criteria

def compute_board_readiness_score(kpis, insights):
    # 1. KPI Confidence (40%)
    kpi_conf = kpis.get("_confidence", {})
    avg_kpi = sum(kpi_conf.values()) / len(kpi_conf) if kpi_conf else 0.6
    s1 = avg_kpi * 40

    # 2. Insight Trust (30%)
    total_w = sum(1 for _ in insights[:8]) or 1
    trusted = sum(1 for i in insights[:8] if i.get("executive_summary_flag"))
    s2 = (trusted / total_w) * 30

    # 3. Coverage (30%)
    req = ["board_confidence_score", "total_volume", "avg_duration", "avg_unit_cost", "variance_score"]
    present = sum(1 for k in req if k in kpis and kpis[k] is not None)
    s3 = (present / len(req)) * 30

    # 4. Penalty
    crit = sum(1 for i in insights if i.get("level") == "CRITICAL")
    penalty = 10 if crit >= 3 else (5 if crit > 0 else 0)

    final = round(max(0, min(100, s1 + s2 + s3 - penalty)))
    
    return {
        "score": final,
        "band": "BOARD READY" if final >= 85 else "REVIEW WITH CAUTION" if final >= 70 else "MANAGEMENT ONLY" if final >= 50 else "NOT BOARD SAFE",
        "components": {"kpi": round(s1,1), "insight": round(s2,1), "coverage": round(s3,1), "penalty": -penalty}
    }

# =====================================================
# MAIN BUILDER (SINGLE SOURCE OF TRUTH)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    
    confidence_map = kpis.get("_confidence", {})
    
    # 1. Rank Insights by Confidence (Prevent false alarms)
    insights = confidence_weight_insights(insights, confidence_map)

    # 2. Select & Rank KPIs
    raw_kpis = select_executive_kpis(kpis)
    primary_kpis = []
    for k in raw_kpis:
        key = k.get("key", k["name"].lower().replace(" ", "_"))
        conf = confidence_map.get(key, 0.6)
        primary_kpis.append({
            "name": k["name"], "value": k["value"], 
            "confidence": round(conf, 2),
            "confidence_label": "High" if conf >= 0.85 else "Moderate" if conf >= 0.70 else "Low"
        })
    
    primary_kpis.sort(key=lambda x: abs(float(x["value"] or 0)) * x["confidence"], reverse=True)
    primary_kpis = primary_kpis[:5]

    # 3. Build Snapshot
    snapshot = build_decision_snapshot(kpis, insights, recommendations)
    
    low_conf = [k["name"] for k in primary_kpis if k["confidence"] < 0.7]
    if low_conf:
        snapshot["confidence_note"] = "⚠️ Data Stability Warning: " + ", ".join(low_conf[:3])

    return {
        "snapshot": snapshot,
        "primary_kpis": primary_kpis,
        "top_problems": extract_top_problems(insights),
        "top_actions": [r.get("action") for r in rank_recommendations(recommendations)[:3]],
        "success_criteria": build_success_criteria(kpis),
        "board_readiness": compute_board_readiness_score(kpis, insights),
        "_executive": {"primary_kpis": primary_kpis, "sub_domain": kpis.get("sub_domain")}
    }
