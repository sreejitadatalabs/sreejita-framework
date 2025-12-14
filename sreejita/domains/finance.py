"""Finance Domain Module - Financial metrics and analysis."""
from typing import Dict, List, Any
import pandas as pd
from .base import BaseDomain


class FinanceDomain(BaseDomain):
    """Finance analytics domain module."""
    
    name = "finance"
    description = "Finance Analytics: P&L, ratios, volatility, forecasting"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate financial data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess financial data."""
        df = df.copy()
        df = df.fillna(0)
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate financial KPIs."""
        kpis = {}
        if "revenue" in df.columns:
            kpis["Total Revenue"] = df["revenue"].sum()
        if "expenses" in df.columns:
            kpis["Total Expenses"] = df["expenses"].sum()
            if "revenue" in df.columns:
                kpis["Net Profit"] = kpis["Total Revenue"] - kpis["Total Expenses"]
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate financial insights."""
        insights = []
        if "Total Revenue" in kpis:
            insights.append(f"Total Revenue: ${kpis['Total Revenue']:,.0f}")
        if "Net Profit" in kpis:
            insights.append(f"Net Profit: ${kpis['Net Profit']:,.0f}")
        return insights
