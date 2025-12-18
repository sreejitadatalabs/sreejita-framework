# =====================================================
# Retail Domain Narrative — v2.9.x
# =====================================================

def get_domain_narrative():
    """
    Defines how the Retail domain should speak to executives.
    No logic. No thresholds. Presentation intent only.
    """
    return {
        "headline": {
            "label": "💰 Revenue Status",
            "kpi": "total_sales",
            "format": "currency",
        },
        "health_summary": {
            "GOOD": "Revenue and margins are operating within healthy thresholds.",
            "WARNING": "Cost pressures are emerging and require attention.",
            "RISK": "Profitability is at risk due to elevated costs.",
        },
        "default_next_step": "Initiate shipping cost optimization audit (5–7 days).",
    }
