"""
Customer Domain Insights
------------------------
Rule-based customer insights.
Retail-parity implementation (dict-based).
"""

from typing import Dict, List, Any
from sreejita.core.decision_snapshot import DecisionSnapshot


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None = None
) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    insights.extend(_volume_insights(kpis))
    insights.extend(_churn_retention_insights(kpis, snapshot))
    insights.extend(_value_behavior_insights(kpis))

    return insights


# ---------------------------------------------------------------------
# Insight Groups
# ---------------------------------------------------------------------

def _volume_insights(kpis: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights = []

    total = _val(kpis, "total_customers")
    new = _val(kpis, "new_customers")
    repeat = _val(kpis, "repeat_customers")

    if total:
        insights.append(_insight(
            title="Established Customer Base",
            description=f"The business currently serves {int(total)} customers.",
            impact="Neutral",
            confidence="High"
        ))

    if new == 0:
        insights.append(_insight(
            title="Customer Acquisition Stalled",
            description="No new customers were acquired in the current period.",
            impact="Negative",
            confidence="High"
        ))

    if total and repeat and repeat / total < 0.30:
        insights.append(_insight(
            title="Weak Repeat Engagement",
            description="Less than 30% of customers are returning.",
            impact="Negative",
            confidence="Medium"
        ))

    return insights


def _churn_retention_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None
) -> List[Dict[str, Any]]:
    insights = []

    churn = _val(kpis, "churn_rate")
    retention = _val(kpis, "retention_rate")

    if churn and churn > 20:
        insights.append(_insight(
            title="Elevated Customer Churn",
            description=f"Customer churn is {churn:.1f}%, above healthy levels.",
            impact="Negative",
            confidence="High"
        ))

    if retention and retention >= 75:
        insights.append(_insight(
            title="Healthy Customer Retention",
            description=f"Retention is strong at {retention:.1f}%.",
            impact="Positive",
            confidence="High"
        ))

    if snapshot and snapshot.has_kpi("churn_rate"):
        prev = snapshot.get_kpi("churn_rate")
        if prev and churn and churn - prev > 5:
            insights.append(_insight(
                title="Customer Churn Increasing",
                description="Churn has increased significantly compared to last period.",
                impact="Negative",
                confidence="Medium"
            ))

    return insights


def _value_behavior_insights(
    kpis: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    insights = []

    avg_value = _val(kpis, "average_customer_value")
    freq = _val(kpis, "purchase_frequency")

    if avg_value and avg_value < 100:
        insights.append(_insight(
            title="Low Average Customer Value",
            description="Customer value is relatively low.",
            impact="Negative",
            confidence="Medium"
        ))

    if freq and freq < 1.5:
        insights.append(_insight(
            title="Infrequent Customer Purchases",
            description="Customers purchase infrequently.",
            impact="Negative",
            confidence="Medium"
        ))

    if avg_value and freq and avg_value >= 500 and freq >= 3:
        insights.append(_insight(
            title="High-Value Engaged Customers",
            description="A high-value, high-engagement customer segment exists.",
            impact="Positive",
            confidence="High"
        ))

    return insights


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _val(kpis: Dict[str, Dict[str, Any]], key: str) -> Any:
    return kpis.get(key, {}).get("value")


def _insight(
    title: str,
    description: str,
    impact: str,
    confidence: str
) -> Dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "impact": impact,
        "confidence": confidence,
        "domain": "customer"
    }
