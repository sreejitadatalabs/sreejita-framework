# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List
from .benchmarks import HEALTHCARE_BENCHMARKS as B

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
    if value is None: return "INFO"
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
    Narrative Engine v6.1 (Contextualized & Quantified)
    """
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []
    
    summary = []
    financial = []
    risks = []
    actions = []
    calculated_savings = None 

    if domain == "healthcare":
        
        # --- 1. CONTEXTUALIZED SCORE ---
        score = kpis.get("board_confidence_score", 50)
        if score < 50:
            summary.append(f"GOVERNANCE ALERT: Score of {score}/100 indicates critical operational risk requiring immediate executive intervention.")
        elif score < 70:
            summary.append(f"Score ({score}/100) indicates operational instability; targeted improvements are necessary.")

        # Context Transparency (Keep this)
        if kpis.get("dataset_shape") == "aggregated_operational":
            summary.append("NOTICE: This assessment is based on aggregated operational data; conclusions reflect broad trends.")

        # --- 2. OPERATIONAL & FINANCIAL IMPACT ---
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients", 0)
        cost_per_day = kpis.get("avg_cost_per_day", 2000)

        if avg_los:
            target = B["avg_los"]["good"]       
            
            if avg_los > target:
                excess_days = avg_los - target
                # Quantify the Impact
                opportunity_loss = excess_days * cost_per_day * total_patients
                loss_str = f"${opportunity_loss/1_000_000:.1f}M" if opportunity_loss > 1_000_000 else f"${opportunity_loss/1_000:.0f}K"
                calculated_savings = loss_str
                
                # Force Impact into Summary
                summary.append(f"FINANCIAL IMPACT: Excess LOS represents a {loss_str} annualized opportunity cost.")
                financial.append(f"Reducing LOS to {target} days would recover {loss_str} in capacity value.")

                if avg_los > B["avg_los"]["critical"]:
                    summary.append(f"CRITICAL: Average LOS ({avg_los:.1f}d) exceeds crisis threshold ({B['avg_los']['critical']}d).")
                else:
                    summary.append(f"Efficiency Gap: Average LOS ({avg_los:.1f}d) underperforms benchmark ({target}d).")
            else:
                summary.append(f"Efficiency: LOS ({avg_los:.1f}d) is performing well against benchmark ({target}d).")

        # --- 3. CLINICAL QUALITY ---
        readm = kpis.get("readmission_rate")
        if readm:
            limit = B["readmission_rate"]["critical"]
            if readm > limit:
                summary.append(f"Quality Alert: Readmission rate ({readm:.1%}) exceeds limit of {limit:.0%}.")
                risks.append(f"High readmission rate ({readm:.1%}) risks value-based payment penalties.")
                actions.append(ActionItem("Implement discharge transition audit", "Nursing Leadership", "Immediate", f"Reduce readmissions < {B['readmission_rate']['good']:.0%}"))

        # --- 4. FINANCIAL HEALTH ---
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", 15000) 
        if avg_cost and avg_cost > benchmark_cost:
             diff = avg_cost - benchmark_cost
             summary.append(f"Financial Alert: Cost per patient (${avg_cost:,.0f}) is ${diff:,.0f} above the approved benchmark.")

        # --- 5. DATA TRUST ---
        comp = kpis.get("data_completeness", 1.0)
        if comp < 0.9:
            risks.append(f"Data Reliability: Key clinical fields are {100-comp*100:.0f}% incomplete.")

        # --- 6. INSIGHT INTEGRATION (Replaces old Root Cause Injection) ---
        for insight in insights:
            # Lift high-priority insights to summary (Root Causes, Quality Gaps)
            if insight['level'] == 'CRITICAL' or insight['title'] == "Quality Blind Spot":
                # Check for duplicates before adding
                msg = f"{insight['title']}: {insight['so_what']}"
                # Simple dedupe check
                if msg not in summary:
                    summary.append(msg)

    # Universal Fallbacks
    if not summary: summary.append("Operational indicators are within expected thresholds.")
    if not financial: 
        if kpis.get("avg_cost_per_patient"): financial.append(f"Current cost structure is sustainable at ${kpis['avg_cost_per_patient']:,.0f}/patient.")
        else: financial.append("No immediate material financial risks detected.")

    # Recommendations Engine
    for rec in recommendations[:5]: 
        if isinstance(rec, dict):
            action_text = rec.get("action", "Review metrics")
            outcome = rec.get("expected_outcome", "Improve KPI")
            timeline = rec.get("timeline", "")
            
            if calculated_savings and any(x in action_text.lower() for x in ["discharge", "los", "pathways"]):
                outcome = f"{outcome} (Est. Impact: {calculated_savings})"

            is_short_term = any(x in timeline.lower() for x in ["30 days", "immediate", "90 days", "q1"])
            prefix = "⚡ [90-DAY FOCUS] " if is_short_term else ""

            actions.append(ActionItem(
                action=f"{prefix}{action_text}",
                owner=rec.get("owner", "Operations"),
                timeline=timeline,
                success_kpi=outcome
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
