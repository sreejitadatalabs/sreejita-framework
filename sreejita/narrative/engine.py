# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List
import math

# =====================================================
# HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
# =====================================================

HEALTHCARE_BENCHMARKS = {
    "avg_los": 5.0,                # days
    "readmission_rate": 0.10,      # 10%
    "long_stay_rate": 0.15,        # 15%
    "bed_turnover_index": 0.20,    # ~1 patient every 5 days
    "cost_multiplier": 2.0,        # vs median
    "provider_variance": 0.30,     # CV
}

HEALTHCARE_THRESHOLDS = {
    "avg_los": {
        "warning": 7.0,
        "critical": 9.0,
    },
    "readmission_rate": {
        "warning": 0.15,
        "critical": 0.20,
    },
    "long_stay_rate": {
        "warning": 0.20,
        "critical": 0.30,
    },
    "provider_variance_score": {
        "warning": 0.40,
        "critical": 0.60,
    },
    "data_completeness": {
        "warning": 0.90,  # Below 0.90 is warning
        "critical": 0.75, # Below 0.75 is critical
    }
}

# =====================================================
# NARRATIVE OUTPUT MODELS
# =====================================================

@dataclass
class ActionItem:
    action: str
    owner: str
    timeline: str
    success_kpi: str

@dataclass
class NarrativeResult:
    executive_summary: List[str]
    financial_impact: List[str]
    risks: List[str]
    action_plan: List[ActionItem]
    key_findings: List[Dict[str, Any]]

# =====================================================
# INTELLIGENCE HELPERS
# =====================================================

def classify_severity(value, metric, thresholds, reverse=False):
    """
    Returns INFO, WARNING, or CRITICAL based on thresholds.
    reverse=True means 'Lower is Better' (e.g. data completeness).
    """
    if value is None:
        return "INFO"
    
    t = thresholds.get(metric, {})
    
    if reverse:
        # For data completeness: < Critical is bad
        if value < t.get("critical", 0): return "CRITICAL"
        if value < t.get("warning", 0): return "WARNING"
    else:
        # For standard metrics: > Critical is bad
        if value >= t.get("critical", float("inf")): return "CRITICAL"
        if value >= t.get("warning", float("inf")): return "WARNING"
        
    return "INFO"

# =====================================================
# MAIN BUILDER
# =====================================================

def build_narrative(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> NarrativeResult:
    """
    Deterministic executive narrative engine (v4.0).
    """
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []
    
    summary: List[str] = []
    risks: List[str] = []
    actions: List[ActionItem] = []
    financial: List[str] = []

    # -------------------------------------------------
    # DOMAIN INTELLIGENCE: HEALTHCARE
    # -------------------------------------------------
    if domain == "healthcare":
        b = HEALTHCARE_BENCHMARKS
        t = HEALTHCARE_THRESHOLDS
        
        # 1. Operational Efficiency (LOS)
        avg_los = kpis.get("avg_los")
        if avg_los:
            sev = classify_severity(avg_los, "avg_los", t)
            if sev == "CRITICAL":
                summary.append(f"Critical operational strain: Average LOS ({avg_los:.1f} days) exceeds crisis threshold ({t['avg_los']['critical']} days), severely impacting bed capacity.")
                risks.append("Severe bed capacity bottleneck detected.")
            elif sev == "WARNING":
                summary.append(f"Operational efficiency warning: LOS ({avg_los:.1f} days) is above the {t['avg_los']['warning']}-day warning level.")
            elif avg_los <= b["avg_los"]:
                summary.append(f"High Clinical Efficiency: Average LOS ({avg_los:.1f} days) is performing better than the industry benchmark ({b['avg_los']} days).")
        else:
            summary.append("Length-of-stay metrics could not be evaluated due to incomplete admission/discharge data, limiting efficiency analysis.")

        # 2. Clinical Quality (Readmission)
        readm = kpis.get("readmission_rate")
        if readm:
            sev = classify_severity(readm, "readmission_rate", t)
            if sev in ["CRITICAL", "WARNING"]:
                summary.append(f"Clinical risk elevated: Readmission rate ({readm:.1%}) exceeds the {b['readmission_rate']:.0%} benchmark, suggesting gaps in discharge planning.")
                risks.append(f"High readmission rate ({readm:.1%}).")
                
                # Auto-inject Action
                actions.append(ActionItem(
                    action="Implement mandatory post-discharge follow-up calls",
                    owner="Nursing Leadership",
                    timeline="Immediate",
                    success_kpi=f"Reduce readmissions to <{b['readmission_rate']:.0%}"
                ))

        # 3. Financial Health
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", 15000)
        
        if avg_cost and avg_cost > benchmark_cost * 1.2:
            summary.append(f"Cost anomaly detected: Average cost per patient (${avg_cost:,.0f}) is significantly above expected baseline.")
            financial.append(f"High cost per episode (${avg_cost:,.0f}) is eroding margin potential.")
        elif avg_cost:
            summary.append(f"Cost efficiency appears stable (${avg_cost:,.0f}/patient), remaining within benchmark tolerance.")

        # 4. Data Trust
        comp = kpis.get("data_completeness", 1.0)
        if classify_severity(comp, "data_completeness", t, reverse=True) != "INFO":
            summary.append(f"Data Completeness Risk: Key clinical fields are {100-comp*100:.0f}% incomplete.")
            risks.append("Low data integrity limits confidence in efficiency conclusions.")

    # -------------------------------------------------
    # FALLBACK & CLEANUP
    # -------------------------------------------------
    
    # Generic visual reinforcement
    if kpis.get("total_patients") and not summary:
        summary.append("Admission volumes indicate stable demand patterns without immediate capacity risks.")

    if not summary:
        summary.append("Operational indicators are within expected thresholds.")

    if not financial:
        financial.append("No immediate material financial risk detected.")

    if not risks:
        risks.append("No critical risks identified at this time.")

    # Process Recommendations (Pass-through + Defaults)
    for rec in recommendations[:3]:
        if isinstance(rec, dict):
            actions.append(ActionItem(
                action=rec.get("action", "Review metrics"),
                owner=rec.get("owner", "Operations"),
                timeline=rec.get("timeline", "Quarterly"),
                success_kpi=rec.get("expected_outcome", "Metric improvement")
            ))

    if not actions:
        actions.append(ActionItem("Monitor key operational metrics", "Ops Lead", "Ongoing", "Stability"))

    return NarrativeResult(
        executive_summary=list(dict.fromkeys(summary)), # Dedupe
        financial_impact=financial,
        risks=risks,
        action_plan=actions,
        key_findings=insights
    )

def generate_narrative(*args, **kwargs):
    return build_narrative(*args, **kwargs)
