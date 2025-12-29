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

def classify_severity(value, metric, thresholds, reverse=False):
    if value is None: return "INFO"
    t = thresholds.get(metric, {})
    # reverse=True means Lower is Bad (e.g. Data Completeness)
    if reverse:
        if value < t.get("critical", 0): return "CRITICAL"
        if value < t.get("warning", 0): return "WARNING"
    # reverse=False means Higher is Bad (e.g. LOS)
    else:
        if value >= t.get("critical", float("inf")): return "CRITICAL"
        if value >= t.get("warning", float("inf")): return "WARNING"
    return "INFO"

def build_narrative(domain: str, kpis: Dict[str, Any], insights: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]) -> NarrativeResult:
    kpis = kpis or {}
    insights = insights or []
    recommendations = recommendations or []
    summary, financial, risks, actions = [], [], [], []

    if domain == "healthcare":
        # 1. Operational & Financial Translation
        avg_los = kpis.get("avg_los")
        total_patients = kpis.get("total_patients", 0)
        # Robust Cost Calculation
        cost_per_day = kpis.get("avg_cost_per_day")
        if not cost_per_day and avg_los and kpis.get("avg_cost_per_patient"):
            cost_per_day = kpis["avg_cost_per_patient"] / avg_los
        if not cost_per_day: cost_per_day = 2000

        if avg_los:
            target = B["avg_los"]["good"]
            limit = B["avg_los"]["critical"]
            
            if avg_los > limit:
                # 💰 THE FINANCIAL TRANSLATION
                excess_days = avg_los - target
                opportunity_loss = excess_days * cost_per_day * total_patients
                loss_str = f"${opportunity_loss/1_000_000:.1f}M" if opportunity_loss > 1_000_000 else f"${opportunity_loss/1_000:.0f}K"
                
                summary.append(f"CRITICAL: Average LOS ({avg_los:.1f} days) exceeds the {B['avg_los']['source']} crisis threshold of {limit} days.")
                financial.append(f"Capacity Bottleneck: Excess stay duration ({excess_days:.1f} days over target) represents an estimated **{loss_str}** in annualized opportunity cost.")
                risks.append("Severe bed capacity constraints eroding margin.")
            elif avg_los > B["avg_los"]["warning"]:
                summary.append(f"Efficiency Warning: LOS ({avg_los:.1f} days) is drifting above the {B['avg_los']['warning']}-day alert level.")
            else:
                summary.append(f"Strong Efficiency: LOS ({avg_los:.1f} days) outperforms the benchmark ({target} days).")

        # 2. Clinical Quality
        readm = kpis.get("readmission_rate")
        if readm and readm > B["readmission_rate"]["critical"]:
            summary.append(f"Quality Alert: Readmission rate ({readm:.1%}) exceeds the {B['readmission_rate']['source']} limit.")
            risks.append(f"High readmission rate ({readm:.1%}) risks value-based payment penalties.")

        # 3. Financial Health
        avg_cost = kpis.get("avg_cost_per_patient")
        if avg_cost and avg_cost > B["cost_benchmark_usd"] * 1.2:
            summary.append("Cost Variance: Average cost per patient is significantly above internal benchmarks.")
        elif avg_cost:
            summary.append(f"Financial Stability: Direct cost per patient (${avg_cost:,.0f}) remains efficiently below benchmark.")

    # Fallbacks
    if not summary: summary.append("Operational indicators are within expected thresholds.")
    if not financial: financial.append("No immediate material financial risks detected.")

    # Recommendations (Top 3)
    for rec in recommendations[:3]:
        if isinstance(rec, dict):
            actions.append(ActionItem(
                action=rec.get("action", "Action"),
                owner=rec.get("owner", "Ops"),
                timeline=rec.get("timeline", "Q1"),
                success_kpi=rec.get("expected_outcome", "Improve KPI")
            ))

    if not actions:
        actions.append(ActionItem("Monitor key metrics", "Ops Lead", "Ongoing", "Stability"))

    return NarrativeResult(executive_summary=list(dict.fromkeys(summary)), financial_impact=financial, risks=risks, action_plan=actions, key_findings=insights)

def generate_narrative(*args, **kwargs):
    return build_narrative(*args, **kwargs)
