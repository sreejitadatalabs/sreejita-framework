"""
Ecommerce Domain KPIs

Computes e-commerce specific performance metrics including:
- Conversion rate
- Cart abandonment rate
- Average Order Value (AOV)
- Customer Acquisition Cost (CAC)
- Lifetime Value (LTV)
- Return on Ad Spend (ROAS)
- ROI
- Cart size metrics
- Checkout completion rate
- Return rate
"""

def compute_ecommerce_kpis(df):
    """
    Compute ecommerce KPIs from transaction data.
    
    Expected columns (flexible - adapts to available data):
    - conversion_rate: percentage of visitors converting
    - cart_abandonment: percentage of carts abandoned
    - aov: average order value
    - cac: customer acquisition cost
    - ltv: lifetime value
    - roas: return on ad spend
    - roi: return on investment
    - cart_size: items per order
    - checkout_completion: percentage completing checkout
    - return_rate: percentage of items returned
    
    Returns:
        dict: KPI values (or None if not computable)
    """
    kpis = {}
    
    # Conversion Rate
    if 'conversion_rate' in df.columns:
        try:
            kpis['conversion_rate'] = round(df['conversion_rate'].mean(), 4)
        except:
            kpis['conversion_rate'] = None
    else:
        kpis['conversion_rate'] = None
    
    # Cart Abandonment Rate
    if 'cart_abandonment' in df.columns:
        try:
            kpis['cart_abandonment_rate'] = round(df['cart_abandonment'].mean(), 4)
        except:
            kpis['cart_abandonment_rate'] = None
    else:
        kpis['cart_abandonment_rate'] = None
    
    # Average Order Value
    if 'aov' in df.columns:
        try:
            kpis['average_order_value'] = round(df['aov'].mean(), 2)
        except:
            kpis['average_order_value'] = None
    else:
        kpis['average_order_value'] = None
    
    # Customer Acquisition Cost
    if 'cac' in df.columns:
        try:
            kpis['customer_acquisition_cost'] = round(df['cac'].mean(), 2)
        except:
            kpis['customer_acquisition_cost'] = None
    else:
        kpis['customer_acquisition_cost'] = None
    
    # Lifetime Value
    if 'ltv' in df.columns:
        try:
            kpis['lifetime_value'] = round(df['ltv'].mean(), 2)
        except:
            kpis['lifetime_value'] = None
    else:
        kpis['lifetime_value'] = None
    
    # Return on Ad Spend
    if 'roas' in df.columns:
        try:
            kpis['return_on_ad_spend'] = round(df['roas'].mean(), 4)
        except:
            kpis['return_on_ad_spend'] = None
    else:
        kpis['return_on_ad_spend'] = None
    
    # ROI
    if 'roi' in df.columns:
        try:
            kpis['roi'] = round(df['roi'].mean(), 4)
        except:
            kpis['roi'] = None
    else:
        kpis['roi'] = None
    
    # Cart Size
    if 'cart_size' in df.columns:
        try:
            kpis['avg_cart_size'] = round(df['cart_size'].mean(), 2)
        except:
            kpis['avg_cart_size'] = None
    else:
        kpis['avg_cart_size'] = None
    
    # Checkout Completion
    if 'checkout_completion' in df.columns:
        try:
            kpis['checkout_completion_rate'] = round(df['checkout_completion'].mean(), 4)
        except:
            kpis['checkout_completion_rate'] = None
    else:
        kpis['checkout_completion_rate'] = None
    
    # Return Rate
    if 'return_rate' in df.columns:
        try:
            kpis['return_rate'] = round(df['return_rate'].mean(), 4)
        except:
            kpis['return_rate'] = None
    else:
        kpis['return_rate'] = None
    
    # Summary metrics
    try:
        kpis['total_transactions'] = len(df)
    except:
        kpis['total_transactions'] = 0
    
    try:
        # Calculate CAC:LTV ratio if both available
        if kpis.get('customer_acquisition_cost') and kpis.get('lifetime_value'):
            cac = kpis['customer_acquisition_cost']
            ltv = kpis['lifetime_value']
            if cac > 0:
                kpis['cac_ltv_ratio'] = round(ltv / cac, 2)
            else:
                kpis['cac_ltv_ratio'] = None
        else:
            kpis['cac_ltv_ratio'] = None
    except:
        kpis['cac_ltv_ratio'] = None
    
    return kpis
