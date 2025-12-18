"""
Ecommerce Domain Insights

Generates business insights from ecommerce KPIs including:
- Conversion funnel health
- Cart abandonment risks
- AOV trends and performance
- Customer acquisition efficiency
- Return/churn risks
- ROAS and profitability analysis
"""

def generate_ecommerce_insights(df, kpis):
    """
    Generate ecommerce-specific insights from data and KPIs.
    
    Args:
        df: pandas DataFrame with transaction data
        kpis: dict of computed KPIs
    
    Returns:
        list: Text insights (can be empty if insufficient data)
    """
    insights = []
    
    if not kpis or all(v is None for v in kpis.values()):
        return ["WARNING: Insufficient ecommerce KPI data for insights."]
    
    # Conversion rate insights
    conv_rate = kpis.get('conversion_rate')
    if conv_rate is not None:
        if conv_rate < 0.01:
            insights.append("CRITICAL: Very low conversion rate ({}%) - investigate checkout friction.".format(conv_rate*100))
        elif conv_rate < 0.03:
            insights.append("WARNING: Below-average conversion ({}%) - consider UX improvements.".format(conv_rate*100))
        else:
            insights.append("SUCCESS: Strong conversion rate ({}%).".format(conv_rate*100))
    
    # Cart abandonment insights
    cart_abandon = kpis.get('cart_abandonment_rate')
    if cart_abandon is not None:
        if cart_abandon > 0.7:
            insights.append("CRITICAL: Cart abandonment ({:.1f}%) - implement recovery strategy.".format(cart_abandon*100))
        elif cart_abandon > 0.5:
            insights.append("WARNING: High cart abandonment ({:.1f}%) - review shipping/payment options.".format(cart_abandon*100))
        else:
            insights.append("SUCCESS: Healthy cart completion rate ({:.1f}%).".format((1-cart_abandon)*100))
    
    # AOV insights
    aov = kpis.get('average_order_value')
    if aov is not None:
        insights.append("INFO: Average Order Value: ${:.2f} - consider upsell/cross-sell opportunities.".format(aov))
    
    # CAC vs LTV insights
    cac_ltv = kpis.get('cac_ltv_ratio')
    if cac_ltv is not None:
        if cac_ltv < 1:
            insights.append("CRITICAL: CAC:LTV ratio {:.2f} - customer lifetime value lower than acquisition cost.".format(cac_ltv))
        elif cac_ltv < 3:
            insights.append("WARNING: CAC:LTV ratio {:.2f} - improvement needed for profitability.".format(cac_ltv))
        else:
            insights.append("SUCCESS: Healthy CAC:LTV ratio {:.2f} - sustainable growth.".format(cac_ltv))
    
    # Checkout completion insights
    checkout_complete = kpis.get('checkout_completion_rate')
    if checkout_complete is not None:
        if checkout_complete < 0.5:
            insights.append("CRITICAL: Checkout completion {:.1f}% - critical UX issue in payment flow.".format(checkout_complete*100))
        elif checkout_complete < 0.8:
            insights.append("WARNING: Checkout completion {:.1f}% - optimize multi-step process.".format(checkout_complete*100))
    
    # Return rate insights
    return_rate = kpis.get('return_rate')
    if return_rate is not None:
        if return_rate > 0.3:
            insights.append("CRITICAL: High return rate ({:.1f}%) - review product quality/fit.".format(return_rate*100))
        elif return_rate > 0.15:
            insights.append("WARNING: Elevated returns ({:.1f}%) - monitor product issues.".format(return_rate*100))
    
    # ROAS insights
    roas = kpis.get('return_on_ad_spend')
    if roas is not None:
        if roas < 1:
            insights.append("CRITICAL: ROAS {:.2f} - ad spend not profitable, review campaigns.".format(roas))
        elif roas < 2:
            insights.append("WARNING: ROAS {:.2f} - room for optimization.".format(roas))
        else:
            insights.append("SUCCESS: ROAS {:.2f} - strong ad performance.".format(roas))
    
    # Cart size insights
    cart_size = kpis.get('avg_cart_size')
    if cart_size is not None:
        if cart_size < 1.5:
            insights.append("INFO: Low cart size ({:.1f} items) - implement bundling strategy.".format(cart_size))
        elif cart_size > 3:
            insights.append("SUCCESS: Strong cart size ({:.1f} items) - effective cross-selling.".format(cart_size))
    
    # Transaction volume insight
    total_txn = kpis.get('total_transactions')
    if total_txn is not None and total_txn < 10:
        insights.append("WARNING: Low transaction volume - results may not be statistically significant.")
    
    return insights if insights else ["No major ecommerce metrics available for insights."]
