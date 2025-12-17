"""
Customer Domain Insights
------------------------
Rule-based insights derived from customer KPIs.
Follows Retail insight grammar and executive standards.
"""

from typing import Dict, List, Any

from sreejita.core.insights import Insight
from sreejita.core.decision_snapshot import DecisionSnapshot


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_customer_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None = None
) -> List[Insight]:
    """
    Generate customer insights from computed KPIs.

    Args:
        kpis: Output of reporting.customer.kpis.compute_customer_kpis
        snapshot: Optional historical snapshot for delta-based insights

    Returns:
        List[Insight]
    """

    insights: List[Insight] = []

    insights.extend(_customer_volume_insights(kpis, snapshot))
    insights.extend(_churn_retention_insights(kpis, snapshot))
    insights.extend(_value_behavior_insights(kpis, snapshot))

    return insights


# ---------------------------------------------------------------------
# Insight Groups
# ---------------------------------------------------------------------

def _customer_volume_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None
) -> List[Insight]:

    insights = []

    total_customers = _val(kpis, "total_customers")
    new_customers = _val(kpis, "new_customers")
    repeat_customers = _val(kpis, "repeat_customers")

    if total_customers is not None and total_customers > 0:
        insights.append(
            Insight(
                title="Customer Base Established",
                description=f"The business currently has {int(total_customers)} active customers.",
                impact="Neutral",
                confidence="High",
                domain="customer"
            )
        )

    if new_customers is not None and new_customers == 0:
        insights.append(
            Insight(
                title="No New Customer Acquisition",
                description="No new customers were acquired in the current period, which may impact future growth.",
                impact="Negative",
                confidence="High",
                domain="customer"
            )
        )

    if (
        repeat_customers is not None
        and total_customers
        and repeat_customers / total_customers < 0.3
    ):
        insights.append(
            Insight(
                title="Low Repeat Customer Ratio",
                description="Less than 30% of customers are returning, indicating weak customer loyalty.",
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

    churn_rate = _val(kpis, "churn_rate")
    retention_rate = _val(kpis, "retention_rate")

    if churn_rate is not None and churn_rate > 20:
        insights.append(
            Insight(
                title="High Customer Churn",
                description=f"Customer churn is at {churn_rate:.1f}%, which is above healthy levels.",
                impact="Negative",
                confidence="High",
                domain="customer"
            )
        )

    if retention_rate is not None and retention_rate >= 75:
        insights.append(
            Insight(
                title="Strong Customer Retention",
                description=f"Retention rate is healthy at {retention_rate:.1f}%, indicating stable customer engagement.",
                impact="Positive",
                confidence="High",
                domain="customer"
            )
        )

    # Snapshot-based trend insight
    if snapshot and snapshot.has_kpi("churn_rate"):
        prev = snapshot.get_kpi("churn_rate")
        if prev is not None and churn_rate is not None:
            delta = churn_rate - prev
            if delta > 5:
                insights.append(
                    Insight(
                        title="Churn Increasing Over Time",
                        description=f"Customer churn increased by {delta:.1f}% compared to the previous period.",
                        impact="Negative",
                        confidence="Medium",
                        domain="customer"
                    )
                )

    return insights


def _value_behavior_insights(
    kpis: Dict[str, Dict[str, Any]],
    snapshot: DecisionSnapshot | None
) -> List[Insight]:

    insights = []

    avg_value = _val(kpis, "average_customer_value")
    purchase_freq = _val(kpis, "purchase_frequency")

    if avg_value is not None and avg_value < 0:
        # Defensive: should never happen, but protects reports
        return insights

    if avg_value is not None and avg_value < 100:
        insights.append(
            Insight(
                title="Low Average Customer Value",
                description="Average customer value is relatively low, suggesting opportunities for upselling or bundling.",
                impact="Negative",
                confidence="Medium",
                domain="customer"
            )
        )

    if purchase_freq is not None and purchase_freq < 1.5:
        insights.append(
            Insight(
                title="Low Purchase Frequency",
                description="Customers purchase infrequently, which may signal weak engagement.",
                impact="Negative",
                confidence="Medium",
                domain="customer"
            )
        )

    if avg_value is not None and purchase_freq is not None:
        if avg_value >= 500 and purchase_freq >= 3:
            insights.append(
                Insight(
                    title="High-Value Engaged Customers Identified",
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
    Safe extractor for KPI raw value.
    """
    return kpis.get(key, {}).get("value")
