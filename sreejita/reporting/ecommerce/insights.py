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
        return ["⚠ Insufficient ecommerce KPI data for insights."]
    
    # Conversion rate insights
    conv_rate = kpis.get('conversion_rate')
    if conv_rate is not None:
        if conv_rate < 0.01:
            insights.append(f"🔴 Very low conversion rate ({conv_rate*100:.2f}%) - investigate checkout friction.")
        elif conv_rate < 0.03:
            insights.append(f"🟡 Below-average conversion ({conv_rate*100:.2f}%) - consider UX improvements.")
        else:
            insights.append(f"✅ Strong conversion rate ({conv_rate*100:.2f}%).")
    
    # Cart abandonment insights
    cart_abandon = kpis.get('cart_abandonment_rate')
    if cart_abandon is not None:
        if cart_abandon > 0.7:
            insights.append(f"🔴 Critical cart abandonment ({cart_abandon*100:.1f}%) - implement recovery strategy.")
        elif cart_abandon > 0.5:
            insights.append(f"🟡 High cart abandonment ({cart_abandon*100:.1f}%) - review shipping/payment options.")
        else:
            insights.append(f"✅ Healthy cart completion rate ({(1-cart_abandon)*100:.1f}%).")
    
    # AOV insights
    aov = kpis.get('average_order_value')
    if aov is not None:
        insights.append(f"📊 Average Order Value: ${aov:.2f} - consider upsell/cross-sell opportunities.")
    
    # CAC vs LTV insights
    cac_ltv = kpis.get('cac_ltv_ratio')
    if cac_ltv is not None:
        if cac_ltv < 1:
            insights.append(f"🔴 CAC:LTV ratio {cac_ltv:.2f} - customer lifetime value lower than acquisition cost.")
        elif cac_ltv < 3:
            insights.append(f"🟡 CAC:LTV ratio {cac_ltv:.2f} - improvement needed for profitability.")
        else:
            insights.append(f"✅ Healthy CAC:LTV ratio {cac_ltv:.2f} - sustainable growth.")
    
    # Checkout completion insights
    checkout_complete = kpis.get('checkout_completion_rate')
    if checkout_complete is not None:
        if checkout_complete < 0.5:
            insights.append(f"🔴 Checkout completion {checkout_complete*100:.1f}% - critical UX issue in payment flow.")
        elif checkout_complete < 0.8:
            insights.append(f"🟡 Checkout completion {checkout_complete*100:.1f}% - optimize multi-step process.")
    
    # Return rate insights
    return_rate = kpis.get('return_rate')
    if return_rate is not None:
        if return_rate > 0.3:
            insights.append(f"🔴 High return rate ({return_rate*100:.1f}%) - review product quality/fit.")
        elif return_rate > 0.15:
            insights.append(f"🟡 Elevated returns ({return_rate*100:.1f}%) - monitor product issues.")
    
    # ROAS insights
    roas = kpis.get('return_on_ad_spend')
    if roas is not None:
        if roas < 1:
            insights.append(f"🔴 ROAS {roas:.2f} - ad spend not profitable, review campaigns.")
        elif roas < 2:
            insights.append(f"🟡 ROAS {roas:.2f} - room for optimization.")
        else:
            insights.append(f"✅ ROAS {roas:.2f} - strong ad performance.")
    
    # Cart size insights
    cart_size = kpis.get('avg_cart_size')
    if cart_size is not None:
        if cart_size < 1.5:
            insights.append(f"📦 Low cart size ({cart_size:.1f} items) - implement bundling strategy.")
        elif cart_size > 3:
            insights.append(f"✅ Strong cart size ({cart_size:.1f} items) - effective cross-selling.")
    
    # Transaction volume insight
    total_txn = kpis.get('total_transactions')
    if total_txn is not None and total_txn < 10:
        insights.append("⚠ Low transaction volume - results may not be statistically significant.")
    
    return insights if insights else ["No major ecommerce metrics available for insights."]
