"""Universal Column Mapping - Fix #3 for retail bias resolution.

This module provides flexible column mapping across all domains,
allowing KPIs and visualizations to work with any column naming convention.
"""

from typing import List, Optional, Set
import pandas as pd


class ColumnMapping:
    """Universal column mapping for flexible domain analysis."""

    # Financial columns
    REVENUE_COLS = {"revenue", "sales", "income", "total_spend", "total_sales", "amount"}
    COST_COLS = {"cost", "expense", "expenses", "total_cost", "spending"}
    PROFIT_COLS = {"profit", "net_income", "margin", "net_profit", "earnings"}

    # Categorical columns
    CATEGORY_COLS = {"category", "department", "unit", "process_stage", "segment", "division", "group"}
    
    # Temporal columns
    DATE_COLS = {"date", "order_date", "transaction_date", "timestamp", "time_period", "period"}
    
    # ID columns
    ID_COLS = {"id", "customer_id", "order_id", "transaction_id", "patient_id", "entity_id"}

    @staticmethod
    def find_column(df_columns: Set[str], mapping_set: Set[str]) -> Optional[str]:
        """Find first matching column from mapping set in dataframe columns."""
        df_cols_lower = {str(c).lower() for c in df_columns}
        for col in mapping_set:
            if col.lower() in df_cols_lower:
                # Return original column name from df
                return next(c for c in df_columns if str(c).lower() == col.lower())
        return None

    @staticmethod
    def auto_detect(df: pd.DataFrame) -> dict:
        """Auto-detect columns based on universal mapping."""
        return {
            "revenue_col": ColumnMapping.find_column(set(df.columns), ColumnMapping.REVENUE_COLS),
            "cost_col": ColumnMapping.find_column(set(df.columns), ColumnMapping.COST_COLS),
            "profit_col": ColumnMapping.find_column(set(df.columns), ColumnMapping.PROFIT_COLS),
            "category_col": ColumnMapping.find_column(set(df.columns), ColumnMapping.CATEGORY_COLS),
            "date_col": ColumnMapping.find_column(set(df.columns), ColumnMapping.DATE_COLS),
            "id_col": ColumnMapping.find_column(set(df.columns), ColumnMapping.ID_COLS),
        }
