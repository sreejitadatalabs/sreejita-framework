from sreejita.reporting.registry import DOMAIN_REPORT_ENGINES


def generate_report_payload(df, decision, policy):
    """
    v2.6 Orchestrator
    """
    domain = decision.selected_domain

    engine = DOMAIN_REPORT_ENGINES.get(domain)
    if not engine:
        return None

    kpis = engine["kpis"](df)
    insights = engine["insights"](df, kpis)
    recommendations = engine["recommendations"](insights)

    return {
        "domain": domain,
        "domain_confidence": decision.confidence,
        "policy_status": policy.status if policy else None,

        "kpis": kpis,
        "insights": insights,
        "recommendations": recommendations
    }
