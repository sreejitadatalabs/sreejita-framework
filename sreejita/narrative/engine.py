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
    Narrative Engine v6.2 (The '10/10' Edition)
    Features: Clinical Justification Logic + 1-Day Projections
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
        
        # --- 1. GOVERNANCE CONTEXT ---
        interpretation = kpis.get("board_confidence_interpretation", "")
        if interpretation:
            summary.append(f"GOVERNANCE ALERT: {interpretation}")
        
        if kpis.get("dataset_shape") == "aggregated_operational":
            summary.append("NOTICE: This assessment is based on aggregated operational data.")

        # --- 2. OPERATIONAL IMPACT & PROJECTION ---
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients", 0)
        cost_per_day = kpis.get("avg_cost_per_day", 2000)

        if avg_los:
            target = B["avg_los"]["good"]       
            
            if avg_los > target:
                excess_days = avg_los - target
                opportunity_loss = excess_days * cost_per_day * total_patients
                loss_str = f"${opportunity_loss/1_000_000:.1f}M" if opportunity_loss > 1_000_000 else f"${opportunity_loss/1_000:.0f}K"
                calculated_savings = loss_str
                
                summary.append(f"FINANCIAL IMPACT: Excess LOS represents a {loss_str} annualized opportunity cost.")
                
                # GAP D: The "What If" Projection
                one_day_val = total_patients * cost_per_day
                one_day_str = f"${one_day_val/1_000_000:.1f}M" if one_day_val > 1_000_000 else f"${one_day_val/1_000:.0f}K"
                financial.append(f"Projection: Reducing average LOS by just 1 day would recover approximately {one_day_str} annually.")

                if avg_los > B["avg_los"]["critical"]:
                    summary.append(f"CRITICAL: Average LOS ({avg_los:.1f}d) exceeds crisis threshold ({B['avg_los']['critical']}d).")
                else:
                    summary.append(f"Efficiency Gap: Average LOS ({avg_los:.1f}d) underperforms benchmark ({target}d).")
            else:
                summary.append(f"Efficiency: LOS ({avg_los:.1f}d) is performing well against benchmark ({target}d).")

        # --- 3. CLINICAL LOGIC BRIDGE (Gap A) ---
        readm = kpis.get("readmission_rate")
        long_stay = kpis.get("long_stay_rate", 0)
        
        if readm is not None:
            limit = B["readmission_rate"]["critical"]
            if readm > limit:
                summary.append(f"Quality Alert: Readmission rate ({readm:.1%}) is high.")
                if long_stay > 0.25:
                    summary.append("INEFFICIENCY INDICATOR: Extended LOS is not offset by improved outcomes, signaling inefficiency rather than care quality.")
            else:
                if long_stay > 0.25:
                    summary.append("CLINICAL CONTEXT: Extended LOS may be clinically justified, as readmission rates remain controlled.")
        else:
            # Explicit Governance Risk for missing data
            summary.append("GOVERNANCE RISK: Outcome Justification Unknown. Extended LOS cannot be justified without readmission data.")

        # --- 4. FINANCIAL HEALTH ---
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", 15000) 
        if avg_cost and avg_cost > benchmark_cost:
             diff = avg_cost - benchmark_cost
             summary.append(f"Financial Alert: Cost per patient (${avg_cost:,.0f}) is ${diff:,.0f} above the approved benchmark.")

        # --- 5. INSIGHT INTEGRATION ---
        for insight in insights:
            if insight['title'] == "Excess Days Breakdown":
                 clean_text = insight['so_what'].replace("<br/>", "; ")
                 summary.append(f"IMPACT MATH: {clean_text}")
            elif insight['level'] == 'CRITICAL':
                msg = f"{insight['title']}: {insight['so_what']}"
                if msg not in summary: summary.append(msg)

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
