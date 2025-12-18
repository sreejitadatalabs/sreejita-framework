"""Base KPI computation with flexible column mapping (Fix #4).

Provides safe computation functions for all domains without hardcoding column names.
"""

from sreejita.domains.column_mapping import ColumnMapping
from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio
from typing import Dict, Any, Optional
import pandas as pd


class FlexibleKPIEngine:
    """Universal KPI engine supporting flexible column names."""
    
    @staticmethod
    def find_column(df: pd.DataFrame, col_names: list) -> Optional[str]:
        """Find first matching column name (case-insensitive)."""
        df_cols_lower = {str(c).lower(): c for c in df.columns}
        for name in col_names:
            if str(name).lower() in df_cols_lower:
                return df_cols_lower[str(name).lower()]
        return None
    
    @staticmethod
    def safe_get_column(df: pd.DataFrame, col_names: list, default_col: Optional[str] = None) -> Optional[str]:
        """Get column with fallback options."""
        col = FlexibleKPIEngine.find_column(df, col_names)
        return col if col else default_col


def compute_kpis_flexible(df: pd.DataFrame, config: Dict[str, list]) -> Dict[str, Any]:
    """Generic KPI computation with flexible column names.
    
    Args:
        df: DataFrame to analyze
        config: Dict mapping KPI names to column name alternatives
               e.g., {'revenue': ['sales', 'revenue', 'amount'], ...}
    
    Returns:
        Dict of computed KPIs
    """
    kpis = {}
    
    # Auto-detect standard columns
    mapping = ColumnMapping.auto_detect(df)
    
    # Custom column detection from config
    cols = {}
    for key, alternatives in config.items():
        cols[key] = FlexibleKPIEngine.find_column(df, alternatives)
    
    # Compute metrics only if columns exist
    if cols.get('revenue'):
        kpis['total_revenue'] = safe_sum(df, cols['revenue'])
    
    if cols.get('cost'):
        kpis['total_cost'] = safe_sum(df, cols['cost'])
    
    if cols.get('profit'):
        kpis['total_profit'] = safe_sum(df, cols['profit'])
    elif cols.get('revenue') and cols.get('cost'):
        kpis['total_profit'] = kpis.get('total_revenue', 0) - kpis.get('total_cost', 0)
    
    if 'total_profit' in kpis and 'total_revenue' in kpis:
        kpis['profit_margin'] = safe_ratio(kpis['total_profit'], kpis['total_revenue'])
    
    if cols.get('quantity'):
        kpis['total_quantity'] = safe_sum(df, cols['quantity'])
        if cols.get('revenue'):
            kpis['avg_per_unit'] = safe_ratio(kpis.get('total_revenue', 0), kpis['total_quantity'])
    
    if cols.get('category'):
        kpis['unique_categories'] = df[cols['category']].nunique()
    
    return kpis
