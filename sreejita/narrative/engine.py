from dataclasses import dataclass
from typing import Dict, Any, List
from .benchmarks import HEALTHCARE_BENCHMARKS as B

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

def build_narrative(
    domain: str,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> NarrativeResult:
    """
    Narrative Engine v6.0 (Strict Financials + 90-Day Focus)
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
        if kpis.get("dataset_shape") == "aggregated_operational":
            summary.append("NOTICE: This assessment is based on aggregated operational data; conclusions reflect broad trends.")

        # --- 2. Operational Efficiency ---
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients", 0)
        cost_per_day = kpis.get("avg_cost_per_day", 2000)

        if avg_los:
            target = B["avg_los"]["good"]       
            critical = B["avg_los"]["critical"] 
            
            if avg_los > target:
                excess_days = avg_los - target
                opportunity_loss = excess_days * cost_per_day * total_patients
                loss_str = f"${opportunity_loss/1_000_000:.1f}M" if opportunity_loss > 1_000_000 else f"${opportunity_loss/1_000:.0f}K"
                calculated_savings = loss_str
                
                financial_msg = f"Efficiency Opportunity: Reducing LOS to benchmark ({target}d) would recover an estimated **{loss_str}** in annualized capacity costs."
                financial.append(financial_msg)

                if avg_los > critical:
                    summary.append(f"CRITICAL: Average LOS ({avg_los:.1f} days) is at crisis levels, exceeding benchmark by {excess_days:.1f} days.")
                    risks.append("Severe bed capacity constraints eroding margin.")
                else:
                    summary.append(f"Efficiency Gap: Average LOS ({avg_los:.1f} days) underperforms the {B['avg_los']['source']} benchmark ({target} days).")
            else:
                summary.append(f"Strong Efficiency: LOS ({avg_los:.1f} days) outperforms the {B['avg_los']['source']} benchmark ({target} days).")

        # --- 3. Clinical Quality ---
        readm = kpis.get("readmission_rate")
        if readm:
            limit = B["readmission_rate"]["critical"]
            if readm > limit:
                summary.append(f"Quality Alert: Readmission rate ({readm:.1%}) exceeds limit of {limit:.0%}.")
                risks.append(f"High readmission rate ({readm:.1%}) risks value-based payment penalties.")
                actions.append(ActionItem("Implement discharge transition audit", "Nursing Leadership", "Immediate", f"Reduce readmissions < {B['readmission_rate']['good']:.0%}"))

        # --- 4. Financial Health (STRICT FIX 1) ---
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost", 15000) 
        
        if avg_cost:
            # STRICT CHECK: Cost > Benchmark is ALWAYS an alert. No 1.2x buffer.
            if avg_cost > benchmark_cost:
                diff = avg_cost - benchmark_cost
                summary.append(f"Financial Alert: Avg cost (${avg_cost:,.0f}) exceeds benchmark (${benchmark_cost:,.0f}).")
                financial.append(f"High cost per episode is eroding margin potential.")
            elif avg_cost < benchmark_cost:
                summary.append(f"Financial Stability: Direct cost per patient (${avg_cost:,.0f}) remains efficiently below benchmark.")

        # --- 5. Data Trust ---
        comp = kpis.get("data_completeness", 1.0)
        if comp < 0.9:
            risks.append(f"Data Reliability: Key clinical fields are {100-comp*100:.0f}% incomplete.")

        # --- 6. Root Cause Injection ---
        for insight in insights:
            if insight['level'] == 'CRITICAL' and 'Root Cause' in insight['title']:
                if "Root Cause Identified" not in str(summary):
                    summary.append(f"{insight['title']}: {insight['so_what']}")

    # Universal Fallbacks
    if not summary: summary.append("Operational indicators are within expected thresholds.")
    if not financial: 
        if kpis.get("avg_cost_per_patient"): financial.append(f"Current cost structure is sustainable at ${kpis['avg_cost_per_patient']:,.0f}/patient.")
        else: financial.append("No immediate material financial risks detected.")

    # Recommendations Engine (90-Day Tagging)
    for rec in recommendations[:5]: 
        if isinstance(rec, dict):
            action_text = rec.get("action", "Review metrics")
            outcome = rec.get("expected_outcome", "Improve KPI")
            timeline = rec.get("timeline", "")
            
            if calculated_savings and any(x in action_text.lower() for x in ["discharge", "los", "length of stay"]):
                outcome = f"{outcome} (Est. Opportunity: {calculated_savings})"

            # ⚡ 90-DAY FOCUS TAG
            is_short_term = any(x in timeline.lower() for x in ["30 days", "immediate", "90 days", "q1", "quarterly"])
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
