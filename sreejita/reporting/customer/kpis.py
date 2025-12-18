from sreejita.reporting.utils import safe_sum, safe_mean
import re

def find_column(df, patterns):
    """Find column matching any of the patterns (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in patterns:
            if re.search(pattern, col_lower):
                return col
    return None

def compute_customer_kpis(df):
    """
    Customer KPIs - Retail-parity contract
    """
    # Flexible column detection
    customer_col = find_column(df, [r'customer.*id', r'cust.*id', r'user.*id', r'client.*id'])
    revenue_col = find_column(df, [r'revenue', r'sales', r'amount', r'total'])
    
    # Safe extraction with defaults
    customers = df[customer_col] if customer_col else df.iloc[:, 0]
    revenue = df[revenue_col] if revenue_col else df.select_dtypes(include='number').iloc[:, 0]
    
    kpis = {
        "total_customers": int(customers.nunique()),
        "total_revenue": float(safe_sum(df, revenue_col) if revenue_col else 0),
        "average_customer_value": float(safe_mean(df, revenue_col) if revenue_col else 0),
    }
    
    return kpis
