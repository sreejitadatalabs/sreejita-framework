"""Retail KPIs with flexible column mapping (Fix #4)."""

from sreejita.domains.column_mapping import ColumnMapping
from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio


def compute_retail_kpis(df, mapping=None):
    """Compute retail KPIs with flexible column mapping.
    
    Supports any naming convention for revenue, cost, discount, quantity columns.
    """
    
    if mapping is None:
        mapping = ColumnMapping.auto_detect(df)
    
    # Auto-detect columns
    revenue_col = mapping.get('revenue_col')
    cost_col = mapping.get('cost_col')
    profit_col = mapping.get('profit_col')
    category_col = mapping.get('category_col')
    
    # Fallback: look for quantity, discount columns
    quantity_col = None
    discount_col = None
    for col in df.columns:
        if str(col).lower() in ['quantity', 'qty', 'unit']:
            quantity_col = col
        if str(col).lower() in ['discount', 'disc', 'discount_amount']:
            discount_col = col
    
    kpis = {}
    
    # Revenue calculations
    if revenue_col:
        kpis["total_revenue"] = safe_sum(df, revenue_col)
    
    # Profit calculations  
    if profit_col:
        kpis["total_profit"] = safe_sum(df, profit_col)
    elif revenue_col and cost_col:
        kpis["total_profit"] = safe_sum(df, revenue_col) - safe_sum(df, cost_col)
    
    # Margin calculation
    if "total_profit" in kpis and "total_revenue" in kpis:
        kpis["profit_margin"] = safe_ratio(kpis["total_profit"], kpis["total_revenue"])
    
    # Discount metrics
    if discount_col:
        kpis["avg_discount"] = safe_mean(df, discount_col)
        kpis["total_discount"] = safe_sum(df, discount_col)
    
    # Quantity metrics
    if quantity_col:
        kpis["total_quantity"] = safe_sum(df, quantity_col)
        if revenue_col:
            kpis["avg_revenue_per_unit"] = safe_ratio(
                safe_sum(df, revenue_col), 
                safe_sum(df, quantity_col)
            )
    
    # Order value (if we have quantity, calculate average)
    if quantity_col and revenue_col:
        kpis["avg_order_value"] = safe_ratio(
            safe_sum(df, revenue_col),
            len(df)
        )
    
    return kpis
