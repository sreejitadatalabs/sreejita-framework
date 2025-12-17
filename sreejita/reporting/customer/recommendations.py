"""
Customer Domain Recommendations
--------------------------------
Actionable, executive-grade recommendations derived from customer insights.
"""

from typing import List

from sreejita.core.insights import Insight
from sreejita.core.recommendations import Recommendation


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_recommendations(
    insights: List[Insight]
) -> List[Recommendation]:
    """
    Generate customer recommendations from insights.

    Args:
        insights: List of Insight objects from customer insights engine

    Returns:
        List[Recommendation]
    """

    recommendations: List[Recommendation] = []

    for insight in insights:
        if insight.title == "Elevated Customer Churn":
            recommendations.append(
                Recommendation(
                    action="Launch targeted win-back campaigns for churn-risk customers.",
                    rationale="High churn indicates disengagement among existing customers.",
                    expected_impact="Reduced churn and stabilized customer base.",
                    priority="High",
                    domain="customer"
                )
            )

        elif insight.title == "Customer Churn Increasing":
            recommendations.append(
                Recommendation(
                    action="Analyze recent customer drop-offs and strengthen retention touchpoints.",
                    rationale="Rising churn suggests weakening engagement trends.",
                    expected_impact="Improved retention and early churn containment.",
                    priority="High",
                    domain="customer"
                )
            )

        elif insight.title == "Weak Repeat Engagement":
            recommendations.append(
                Recommendation(
                    action="Introduce loyalty incentives to encourage repeat purchases.",
                    rationale="Low repeat engagement reduces lifetime customer value.",
                    expected_impact="Higher purchase frequency and improved customer loyalty.",
                    priority="Medium",
                    domain="customer"
                )
            )

        elif insight.title == "Customer Acquisition Stalled":
            recommendations.append(
                Recommendation(
                    action="Review acquisition channels and optimize onboarding experiences.",
                    rationale="Lack of new customers limits growth momentum.",
                    expected_impact="Increased new customer acquisition rates.",
                    priority="Medium",
                    domain="customer"
                )
            )

        elif insight.title == "Low Average Customer Value":
            recommendations.append(
                Recommendation(
                    action="Develop upsell and cross-sell strategies for existing customers.",
                    rationale="Low customer value indicates under-monetization.",
                    expected_impact="Improved average revenue per customer.",
                    priority="Medium",
                    domain="customer"
                )
            )

        elif insight.title == "Infrequent Customer Purchases":
            recommendations.append(
                Recommendation(
                    action="Increase engagement through personalized offers and reminders.",
                    rationale="Infrequent purchasing signals low ongoing engagement.",
                    expected_impact="Higher purchase frequency and engagement.",
                    priority="Low",
                    domain="customer"
                )
            )

        elif insight.title == "High-Value Engaged Customer Segment":
            recommendations.append(
                Recommendation(
                    action="Create exclusive programs for high-value, loyal customers.",
                    rationale="High-value customers drive disproportionate business impact.",
                    expected_impact="Improved retention and advocacy among top customers.",
                    priority="High",
                    domain="customer"
                )
            )

    return recommendations
