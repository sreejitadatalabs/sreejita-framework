"""
Ecommerce Domain Narrative

Provides domain-specific business context and narrative framework
for ecommerce analysis reports.
"""

def get_domain_narrative():
    """
    Return ecommerce domain narrative context.
    
    Returns:
        dict: Domain narrative metadata
    """
    return {
        "domain": "ecommerce",
        "description": "E-commerce Domain Analysis",
        "context": "Online retail transaction and customer behavior analysis",
        "kpi_categories": {
            "Funnel Metrics": ["conversion_rate", "cart_abandonment_rate", "checkout_completion_rate"],
            "Value Metrics": ["average_order_value", "lifetime_value", "customer_acquisition_cost"],
            "Performance Metrics": ["return_on_ad_spend", "return_on_investment", "cac_ltv_ratio"],
            "Behavior Metrics": ["avg_cart_size", "return_rate", "total_transactions"]
        },
        "executive_summary": "E-commerce platform performance driven by conversion rate, cart abandonment, and customer lifetime value metrics.",
        "key_drivers": [
            "Conversion Rate - percentage of visitors making purchases",
            "Cart Abandonment - lost revenue from incomplete transactions",
            "Average Order Value - revenue per transaction",
            "Customer Acquisition Cost - marketing efficiency",
            "Lifetime Value - long-term customer profitability",
            "ROAS - advertising spend efficiency"
        ],
        "risk_factors": [
            "High cart abandonment rates reduce revenue",
            "Low conversion rates indicate funnel friction",
            "High CAC vs LTV ratio threatens sustainability",
            "Elevated return rates impact profitability",
            "Poor ROAS wastes marketing budget"
        ],
        "opportunities": [
            "Optimize checkout flow to reduce cart abandonment",
            "Implement cart recovery email campaigns",
            "Test pricing and bundling strategies",
            "Improve product recommendations (cross-sell/upsell)",
            "Refine ad targeting to improve CAC",
            "Analyze return patterns for quality issues"
        ]
    }

