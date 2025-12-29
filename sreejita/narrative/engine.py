# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List

# =====================================================
# HEALTHCARE BENCHMARKS & THRESHOLDS (AUTHORITATIVE)
# =====================================================

HEALTHCARE_BENCHMARKS = {
    "avg_los": 5.0,                # Target: 5.0 days (Acute care standard)
    "readmission_rate": 0.10,      # Target: 10% (CMS benchmark)
    "bed_turnover_index": 0.20,    # Target: 0.20 (Efficiency standard)
    "cost_benchmark_usd": 15000,   # Global baseline (approximate)
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
    reverse=True: Lower is better (e.g. Completeness)
    reverse=False: Higher is worse (e.g. LOS, Cost)
    """
    if value is None:
        return "INFO"
    
    t = thresholds.get(metric, {})
    
    if reverse:
        # Lower is Bad (e.g. Data Completeness)
        if value < t.get("critical", 0): return "CRITICAL"
        if value < t.get("warning", 0): return "WARNING"
    else:
        # Higher is Bad (e.g. LOS)
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
    Deterministic Executive Narrative Engine v4.0
    Generates judgment, financial impact, and strategic actions.
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
        
        # --- 1. Operational Efficiency (LOS) ---
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients", 0)
        
        # Robust Cost Calculation (Critical for Financial Translation)
        # 1. Try explicit KPI -> 2. Derive from Total/LOS -> 3. Fallback
        cost_per_day = kpis.get("avg_cost_per_day")
        if not cost_per_day and avg_los and kpis.get("avg_cost_per_patient"):
            cost_per_day = kpis["avg_cost_per_patient"] / avg_los
        if not cost_per_day:
            cost_per_day = 2000 # Safe fallback

        if avg_los:
            sev = classify_severity(avg_los, "avg_los", t)
            
            if sev == "CRITICAL":
                summary.append(f"Critical capacity strain: Average LOS ({avg_los:.1f} days) exceeds the crisis threshold of {t['avg_los']['critical']} days.")
                risks.append("Severe bed capacity bottleneck reducing hospital throughput.")
                
                # 💰 FINANCIAL TRANSLATION (The 10/10 Feature)
                excess_days = avg_los - b["avg_los"]
                if excess_days > 0 and total_patients > 0:
                    waste = excess_days * cost_per_day * total_patients
                    waste_str = f"${waste/1_000_000:.1f}M" if waste > 1_000_000 else f"${waste/1_000:.0f}K"
                    financial.append(f"Excess length of stay ({excess_days:.1f} days above target) creates an estimated **{waste_str}** annualized opportunity cost.")

            elif sev == "WARNING":
                summary.append(f"Efficiency warning: LOS ({avg_los:.1f} days) is drifting above the {t['avg_los']['warning']}-day warning level.")
            
            elif avg_los <= b["avg_los"]:
                summary.append(f"High Clinical Efficiency: Average LOS ({avg_los:.1f} days) is performing better than the {b['avg_los']}-day benchmark.")
        else:
            summary.append("Length-of-stay efficiency could not be evaluated due to incomplete admission/discharge data.")

        # --- 2. Clinical Quality (Readmission) ---
        readm = kpis.get("readmission_rate")
        if readm:
            sev = classify_severity(readm, "readmission_rate", t)
            if sev in ["CRITICAL", "WARNING"]:
                summary.append(f"Quality alert: Readmission rate ({readm:.1%}) is above the {b['readmission_rate']:.0%} target, indicating premature discharge risks.")
                risks.append(f"Elevated readmission rate ({readm:.1%}) risks payer penalties.")
                
                # Auto-Action
                actions.append(ActionItem(
                    action="Implement mandatory post-discharge follow-up calls", 
                    owner="Nursing Leadership", 
                    timeline="Immediate", 
                    success_kpi=f"Reduce readmissions < {b['readmission_rate']:.0%}"
                ))

        # --- 3. Financial Health (Cost Judgment) ---
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", b["cost_benchmark_usd"])
        
        if avg_cost:
            if avg_cost > benchmark_cost * 1.2:
                summary.append(f"Financial variance: Avg cost per patient (${avg_cost:,.0f}) is significantly above the benchmark (${benchmark_cost:,.0f}).")
                financial.append(f"High cost per episode is eroding margin potential.")
            elif avg_cost < benchmark_cost:
                summary.append(f"Cost efficiency is stable, with avg cost (${avg_cost:,.0f}) remaining below benchmark.")

        # --- 4. Data Trust Judgment ---
        comp = kpis.get("data_completeness", 1.0)
        sev = classify_severity(comp, "data_completeness", t, reverse=True)
        if sev != "INFO":
            summary.append(f"Data Reliability Warning: Key clinical fields are {100-comp*100:.0f}% incomplete.")
            risks.append("Low data integrity limits confidence in efficiency conclusions.")

    # -------------------------------------------------
    # FALLBACKS & CLEANUP
    # -------------------------------------------------
    
    # Generic visual reinforcement
    if kpis.get("total_patients") and not summary:
        summary.append("Admission volumes indicate stable demand patterns.")
    
    if not summary: summary.append("Operational indicators are within expected thresholds.")
    if not financial: financial.append("No immediate material financial risk detected.")
    if not risks: risks.append("No critical risks identified at this time.")

    # Process Recommendations (Pass-through + Defaults)
    # We take top 3 recommendations from domain and ensure they have executive fields
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

    # Return final result
    return NarrativeResult(
        executive_summary=list(dict.fromkeys(summary)), # Dedupe
        financial_impact=financial,
        risks=risks,
        action_plan=actions,
        key_findings=insights
    )

def generate_narrative(*args, **kwargs):
    return build_narrative(*args, **kwargs)
