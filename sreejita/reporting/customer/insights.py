"""
Customer Domain Insights
------------------------
Rule-based, executive-grade insights derived from customer KPIs.
Retail-style grammar, customer semantics.
"""

from typing import Dict, List, Any

from sreejita.core.insights import Insight
from sreejita.core.decision_snapshot import DecisionSnapshot


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None = None
) -> List[Insight]:
    """
    Generate customer insights from KPI outputs.

    Args:
        kpis: Output from reporting.customer.kpis.compute_customer_kpis
        snapshot: Optional DecisionSnapshot for trend-based insights

    Returns:
        List[Insight]
    """

    insights: List[Insight] = []

    insights.extend(_volume_insights(kpis))
    insights.extend(_churn_retention_insights(kpis, snapshot))
    insights.extend(_value_behavior_insights(kpis))

    return insights


# ---------------------------------------------------------------------
# Insight Groups
# ---------------------------------------------------------------------

def _volume_insights(kpis: Dict[str, Dict[str, Any]]) -> List[Insight]:
    insights = []

    total = _val(kpis, "total_customers")
    new = _val(kpis, "new_customers")
    repeat = _val(kpis, "repeat_customers")

    if total is not None and total > 0:
        insights.append(
            Insight(
                title="Established Customer Base",
                description=f"The business currently serves {int(total)} customers.",
                impact="Neutral",
                confidence="High",
                domain="customer"
            )
        )

    if new is not None and new == 0:
        insights.append(
            Insight(
                title="Customer Acquisition Stalled",
                description="No new customers were acquired in the current period, which may constrain future growth.",
                impact="Negative",
                confidence="High",
                domain="customer"
            )
        )

    if (
        repeat is not None
        and total
        and repeat / total < 0.30
    ):
        insights.append(
            Insight(
                title="Weak Repeat Engagement",
                description="Less than 30% of customers are returning, indicating low repeat engagement.",
                impact="Negative",
                confidence="Medium",
                domain="customer"
            )
        )

    return insights


def _churn_retention_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None
) -> List[Insight]:
    insights = []

    churn = _val(kpis, "churn_rate")
    retention = _val(kpis, "retention_rate")

    if churn is not None and churn > 20:
        insights.append(
            Insight(
                title="Elevated Customer Churn",
                description=f"Customer churn is {churn:.1f}%, exceeding healthy thresholds.",
                impact="Negative",
                confidence="High",
                domain="customer"
            )
        )

    if retention is not None and retention >= 75:
        insights.append(
            Insight(
                title="Healthy Customer Retention",
                description=f"Retention remains strong at {retention:.1f}%, indicating stable engagement.",
                impact="Positive",
                confidence="High",
                domain="customer"
            )
        )

    # Trend-based insight (snapshot-aware)
    if snapshot and snapshot.has_kpi("churn_rate"):
        previous = snapshot.get_kpi("churn_rate")
        if previous is not None and churn is not None:
            delta = churn - previous
            if delta > 5:
                insights.append(
                    Insight(
                        title="Customer Churn Increasing",
                        description=f"Churn increased by {delta:.1f}% compared to the previous period.",
                        impact="Negative",
                        confidence="Medium",
                        domain="customer"
                    )
                )

    return insights


def _value_behavior_insights(
    kpis: Dict[str, Dict[str, Any]]
) -> List[Insight]:
    insights = []

    avg_value = _val(kpis, "average_customer_value")
    frequency = _val(kpis, "purchase_frequency")

    if avg_value is not None and avg_value < 100:
        insights.append(
            Insight(
                title="Low Average Customer Value",
                description="Average customer value is low, suggesting opportunities for upselling or bundling.",
                impact="Negative",
                confidence="Medium",
                domain="customer"
            )
        )

    if frequency is not None and frequency < 1.5:
        insights.append(
            Insight(
                title="Infrequent Customer Purchases",
                description="Customers are purchasing infrequently, indicating weak ongoing engagement.",
                impact="Negative",
                confidence="Medium",
                domain="customer"
            )
        )

    if (
        avg_value is not None
        and frequency is not None
        and avg_value >= 500
        and frequency >= 3
    ):
        insights.append(
            Insight(
                title="High-Value Engaged Customer Segment",
                description="A segment of customers shows both high value and frequent engagement.",
                impact="Positive",
                confidence="High",
                domain="customer"
            )
        )

    return insights


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _val(kpis: Dict[str, Dict[str, Any]], key: str) -> Any:
    """
    Safe extractor for raw KPI values.
    """
    return kpis.get(key, {}).get("value")
