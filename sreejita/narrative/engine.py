# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List

# =====================================================
# HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
# =====================================================

HEALTHCARE_BENCHMARKS = {
    "avg_los": 5.0,                # Target: 5.0 days
    "readmission_rate": 0.10,      # Target: 10%
    "bed_turnover_index": 0.20,    # Target: 0.20
    "cost_benchmark_usd": 15000,   # Global baseline
}

HEALTHCARE_THRESHOLDS = {
    "avg_los": {
        "warning": 6.0,
        "critical": 8.0,
    },
    "readmission_rate": {
        "warning": 0.12,
        "critical": 0.18,
    },
    "long_stay_rate": {
        "warning": 0.15,
        "critical": 0.25,
    },
    "provider_variance_score": {
        "warning": 0.40,
        "critical": 0.60,
    },
    "data_completeness": {
        "warning": 0.90,  # Below 0.90 is WARNING
        "critical": 0.75, # Below 0.75 is CRITICAL
    }
}

# =====================================================
# OUTPUT MODELS
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
# INTELLIGENCE LOGIC
# =====================================================

def classify_severity(value, metric, thresholds, reverse=False):
    """
    Classifies a value as INFO, WARNING, or CRITICAL.
    """
    if value is None:
        return "INFO"
    
    t = thresholds.get(metric, {})
    
    if reverse:
        if value < t.get("critical", 0): return "CRITICAL"
        if value < t.get("warning", 0): return "WARNING"
    else:
        if value >= t.get("critical", float("inf")): return "CRITICAL"
        if value >= t.get("warning", float("inf")): return "WARNING"
        
    return "INFO"

def build_narrative(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> NarrativeResult:
    """
    Deterministic Narrative Engine v4.0
    """
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []
    
    summary = []
    financial = []
    risks = []
    actions = []

    # -------------------------------------------------
    # DOMAIN INTELLIGENCE: HEALTHCARE
    # -------------------------------------------------
    if domain == "healthcare":
        b = HEALTHCARE_BENCHMARKS
        t = HEALTHCARE_THRESHOLDS
        
        # 1. Operational Efficiency (LOS Judgment)
        avg_los = kpis.get("avg_los")
        if avg_los:
            sev = classify_severity(avg_los, "avg_los", t)
            if sev == "CRITICAL":
                summary.append(f"Critical capacity strain: Average LOS ({avg_los:.1f} days) exceeds the crisis threshold of {t['avg_los']['critical']} days.")
                risks.append("Severe bed bottlenecks reducing hospital throughput.")
            elif sev == "WARNING":
                summary.append(f"Efficiency warning: LOS ({avg_los:.1f} days) is drifting above the {t['avg_los']['warning']}-day warning level.")
            elif avg_los <= b["avg_los"]:
                summary.append(f"High Clinical Efficiency: Average LOS ({avg_los:.1f} days) is performing better than the {b['avg_los']}-day benchmark.")
        else:
            summary.append("Length-of-stay efficiency could not be evaluated due to missing admission/discharge dates.")

        # 2. Clinical Quality (Readmission Judgment)
        readm = kpis.get("readmission_rate")
        if readm:
            sev = classify_severity(readm, "readmission_rate", t)
            if sev in ["CRITICAL", "WARNING"]:
                summary.append(f"Quality alert: Readmission rate ({readm:.1%}) is above the {b['readmission_rate']:.0%} target, indicating premature discharge risks.")
                risks.append(f"Elevated readmission rate ({readm:.1%}).")
                actions.append(ActionItem("Implement post-discharge follow-up calls", "Nursing Leadership", "Immediate", f"Reduce readmissions < {b['readmission_rate']:.0%}"))

        # 3. Financial Health (Cost Judgment)
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", b["cost_benchmark_usd"])
        
        if avg_cost:
            if avg_cost > benchmark_cost * 1.2:
                summary.append(f"Financial variance: Avg cost per patient (${avg_cost:,.0f}) is significantly above the benchmark (${benchmark_cost:,.0f}).")
                financial.append(f"High cost per episode is eroding margin potential.")
            elif avg_cost < benchmark_cost:
                summary.append(f"Cost efficiency is stable, with avg cost (${avg_cost:,.0f}) remaining below benchmark.")

        # 4. Data Trust Judgment
        comp = kpis.get("data_completeness", 1.0)
        sev = classify_severity(comp, "data_completeness", t, reverse=True)
        if sev != "INFO":
            summary.append(f"Data Reliability Warning: Key clinical fields are {100-comp*100:.0f}% incomplete.")
            risks.append("Low data integrity limits confidence in efficiency conclusions.")

    # -------------------------------------------------
    # FALLBACKS
    # -------------------------------------------------
    if kpis.get("total_patients") and not summary:
        summary.append("Admission volumes indicate stable demand patterns.")
    
    if not summary: summary.append("Operational indicators are within expected thresholds.")
    if not financial: financial.append("No immediate material financial risk detected.")
    if not risks: risks.append("No critical risks identified at this time.")

    # Process Recommendations (Pass-through)
    # Filter to Top 3 for Report, but engine received 7+
    for rec in recommendations[:3]:
        if isinstance(rec, dict):
            actions.append(ActionItem(
                action=rec.get("action", "Review metrics"),
                owner=rec.get("owner", "Operations"),
                timeline=rec.get("timeline", "Quarterly"),
                success_kpi=rec.get("expected_outcome", "Improve KPI")
            ))

    if not actions:
        actions.append(ActionItem("Monitor key metrics", "Ops Lead", "Ongoing", "Stability"))

    return NarrativeResult(
        executive_summary=list(dict.fromkeys(summary)),
        financial_impact=financial,
        risks=risks,
        action_plan=actions,
        key_findings=insights
    )

def generate_narrative(*args, **kwargs):
    return build_narrative(*args, **kwargs)
