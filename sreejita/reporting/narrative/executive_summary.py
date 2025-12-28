def generate_executive_summary(domain, kpis, insights, recommendations):
    """
    Generates CEO-level narrative (plain English).
    """

    risk = next((i for i in insights if i["level"] in ("CRITICAL", "RISK", "WARNING")), None)
    action = recommendations[0]["action"] if recommendations else "Continue monitoring."

    summary = f"""
Overall, the {domain} operation is functional, but emerging risks require attention.
"""

    if risk:
        summary += f"""
Key Concern: {risk['title']} — {risk['so_what']}
"""

    summary += f"""
Recommended Next Step: {action}
"""

    return summary.strip()
