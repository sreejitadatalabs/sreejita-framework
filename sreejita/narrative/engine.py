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
    Deterministic Executive Narrative Engine v5.1 (Logic & Financial Fix)
    """
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []
    
    summary = []
    financial = []
    risks = []
    actions = []
    
    # Context variable to hold calculated savings for injection into recommendations
    calculated_savings = None 

    # -------------------------------------------------
    # DOMAIN INTELLIGENCE: HEALTHCARE
    # -------------------------------------------------
    if domain == "healthcare":
        
        # 1. Context Transparency
        if kpis.get("dataset_shape") == "aggregated_operational":
            summary.append(
                "NOTICE: This assessment is based on aggregated operational data; conclusions reflect broad trends rather than individual patient journeys."
            )

        # --- 2. Operational Efficiency & Financial Translation ---
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients", 0)
        
        # Robust Cost Calculation
        cost_per_day = kpis.get("avg_cost_per_day")
        if not cost_per_day and avg_los and kpis.get("avg_cost_per_patient"):
            cost_per_day = kpis["avg_cost_per_patient"] / avg_los
        if not cost_per_day:
            cost_per_day = 2000 # Safe fallback

        if avg_los:
            target = B["avg_los"]["good"]       # 5.0
            critical = B["avg_los"]["critical"] # 9.0
            
            # 💰 UNIVERSAL FINANCIAL CALCULATION
            # Calculate impact if we are above TARGET (5.0), not just Critical (9.0)
            if avg_los > target:
                excess_days = avg_los - target
                opportunity_loss = excess_days * cost_per_day * total_patients
                loss_str = f"${opportunity_loss/1_000_000:.1f}M" if opportunity_loss > 1_000_000 else f"${opportunity_loss/1_000:.0f}K"
                
                # SAVE THIS VALUE FOR INJECTION
                calculated_savings = loss_str
                
                financial_msg = f"Efficiency Opportunity: Reducing LOS to benchmark ({target}d) would recover an estimated **{loss_str}** in annualized capacity costs."
                financial.append(financial_msg)

                # NARRATIVE LOGIC FIX
                # Separate Critical Crisis from Standard Inefficiency
                if avg_los > critical:
                    summary.append(f"CRITICAL: Average LOS ({avg_los:.1f} days) is at crisis levels, exceeding benchmark by {excess_days:.1f} days.")
                    risks.append("Severe bed capacity constraints eroding margin.")
                else:
                    # This captures the 6.3 vs 5.0 case correctly now
                    summary.append(f"Efficiency Gap: Average LOS ({avg_los:.1f} days) underperforms the {B['avg_los']['source']} benchmark ({target} days).")
            
            else:
                summary.append(f"Strong Efficiency: LOS ({avg_los:.1f} days) outperforms the {B['avg_los']['source']} benchmark ({target} days).")

        # --- 3. Clinical Quality (Readmission) ---
        readm = kpis.get("readmission_rate")
        if readm:
            limit = B["readmission_rate"]["critical"]
            if readm > limit:
                summary.append(f"Quality Alert: Readmission rate ({readm:.1%}) exceeds the {B['readmission_rate']['source']} limit of {limit:.0%}.")
                risks.append(f"High readmission rate ({readm:.1%}) risks value-based payment penalties.")
                actions.append(ActionItem("Implement discharge transition audit", "Nursing Leadership", "Immediate", f"Reduce readmissions < {B['readmission_rate']['good']:.0%}"))

        # --- 4. Financial Health ---
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", 15000) 
        
        if avg_cost:
            if avg_cost > benchmark_cost * 1.2:
                summary.append(f"Cost Variance: Avg cost per patient is significantly above benchmark (${benchmark_cost:,.0f}).")
            elif avg_cost < benchmark_cost:
                summary.append(f"Financial Stability: Direct cost per patient (${avg_cost:,.0f}) remains efficiently below benchmark.")

        # --- 5. Data Trust ---
        comp = kpis.get("data_completeness", 1.0)
        if comp < 0.9:
            risks.append(f"Data Reliability: Key clinical fields are {100-comp*100:.0f}% incomplete, limiting precision.")

        # --- 6. Root Cause Injection ---
        # If we found specific insights in healthcare.py, lift them to the summary
        for insight in insights:
            if insight['level'] == 'CRITICAL' and 'Bottleneck' in insight['title']:
                # Avoid duplicates
                if "Severe Discharge Bottleneck" not in str(summary):
                    summary.append(f"Root Cause: {insight['so_what']}")

    # -------------------------------------------------
    # UNIVERSAL FALLBACKS
    # -------------------------------------------------
    if not summary:
        summary.append("Operational indicators are within expected thresholds.")
    
    if not financial:
        if kpis.get("avg_cost_per_patient"):
             financial.append(f"Current cost structure is sustainable at ${kpis['avg_cost_per_patient']:,.0f}/patient.")
        else:
             financial.append("No immediate material financial risks detected.")

    # Process Recommendations (INTELLIGENT INJECTION)
    for rec in recommendations[:5]: # Increase limit to 5
        if isinstance(rec, dict):
            action_text = rec.get("action", "Review metrics")
            outcome = rec.get("expected_outcome", "Improve KPI")
            
            # 🔥 INJECT FINANCIAL CONTEXT
            # If we calculated a savings opportunity, and this action is about discharge/LOS, attach the $$$
            if calculated_savings and any(x in action_text.lower() for x in ["discharge", "los", "length of stay"]):
                outcome = f"{outcome} (Est. Opportunity: {calculated_savings})"

            actions.append(ActionItem(
                action=action_text,
                owner=rec.get("owner", "Operations"),
                timeline=rec.get("timeline", "Quarterly"),
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
