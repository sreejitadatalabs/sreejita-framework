def generate_ecommerce_recommendations(df, kpis, insights):
    """Generate actionable recommendations"""
    recommendations = []
    
    if not kpis:
        return ["Insufficient data for recommendations"]
    
    # Conversion optimization
    if kpis.get('conversion_rate', 0) < 0.02:
        recommendations.append("Conduct A/B testing on landing pages to improve conversion")
    
    # Cart abandonment
    if kpis.get('cart_abandonment_rate', 0) > 0.5:
        recommendations.append("Implement one-click checkout and guest checkout option")
        recommendations.append("Create cart recovery email sequence for abandoned carts")
    
    # AOV optimization
    if kpis.get('avg_cart_size', 0) < 2:
        recommendations.append("Implement product bundling and recommended items at checkout")
    
    # CAC optimization
    if kpis.get('customer_acquisition_cost'):
        recommendations.append("Analyze top-performing acquisition channels and allocate budget accordingly")
    
    # Return rate
    if kpis.get('return_rate', 0) > 0.2:
        recommendations.append("Review product descriptions, images, and sizing guides")
        recommendations.append("Analyze return reasons to improve product-market fit")
    
    return recommendations if recommendations else ["Performance is healthy - maintain current strategy"]
