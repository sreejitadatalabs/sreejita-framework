# =====================================================
# Customer Domain Narrative — v2.9.x
# =====================================================

def get_domain_narrative():
    """
    Defines how the Customer domain should speak to executives.
    """
    return {
        "headline": {
            "label": "👥 Customer Base",
            "kpi": "total_customers",
            "format": "count",
        },
        "health_summary": {
            "GOOD": "Customer engagement and value are strong.",
            "WARNING": "Customer engagement shows early signs of decline.",
            "RISK": "Customer churn poses a growth risk.",
        },
        "default_next_step": "Launch targeted retention and loyalty initiatives.",
    }
