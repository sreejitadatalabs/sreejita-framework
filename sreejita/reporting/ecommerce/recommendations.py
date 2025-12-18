"""
Ecommerce Domain Recommendations

Generates actionable recommendations based on ecommerce KPIs.
"""

def generate_ecommerce_recommendations(df, kpis, insights):
    """Generate actionable ecommerce recommendations"""
    recommendations = []
    
    if not kpis or all(v is None for v in kpis.values()):
        return ["Insufficient data for recommendations"]
    
    # Conversion optimization
    conv_rate = kpis.get('conversion_rate', 0)
    if conv_rate < 0.02:
        recommendations.append("Conduct A/B testing on landing pages to improve conversion rate")
        recommendations.append("Ensure mobile-responsive checkout experience")
    
    # Cart abandonment recovery
    cart_abandon = kpis.get('cart_abandonment_rate', 0)
    if cart_abandon > 0.5:
        recommendations.append("Implement one-click checkout and guest checkout option")
        recommendations.append("Create cart recovery email sequence for abandoned carts")
        recommendations.append("Offer multiple payment methods and express checkout")
    
    # AOV & Cart size optimization
    cart_size = kpis.get('avg_cart_size', 0)
    if cart_size < 2:
        recommendations.append("Implement product bundling and frequently bought together recommendations")
        recommendations.append("Add upsell/cross-sell at cart and checkout stages")
    
    # CAC optimization
    cac = kpis.get('customer_acquisition_cost')
    if cac:
        recommendations.append("Analyze top-performing acquisition channels and allocate budget accordingly")
        recommendations.append("Monitor customer acquisition cost trends weekly")
    
    # Return rate management
    return_rate = kpis.get('return_rate', 0)
    if return_rate > 0.2:
        recommendations.append("Review product descriptions, images, and sizing guides")
        recommendations.append("Analyze return reasons (defect, fit, expectation mismatch)")
        recommendations.append("Implement pre-purchase Q&A and customer reviews")
    
    # ROAS optimization
    roas = kpis.get('return_on_ad_spend', 0)
    if roas and roas < 2:
        recommendations.append("Test ad creatives, audiences, and bidding strategies")
        recommendations.append("Implement retargeting campaigns for high-intent users")
        recommendations.append("Use dynamic product ads based on browsing history")
    
    # LTV optimization
    ltv = kpis.get('lifetime_value')
    if ltv:
        recommendations.append("Build loyalty program to increase repeat purchases")
        recommendations.append("Implement email marketing sequences for retention")
    
    # CAC:LTV ratio
    cac_ltv = kpis.get('cac_ltv_ratio')
    if cac_ltv and cac_ltv < 3:
        recommendations.append("CAC:LTV ratio is low - focus on improving customer retention and repeat purchases")
    
    return recommendations if recommendations else ["Performance is strong - maintain current strategy and test new opportunities"]
