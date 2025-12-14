"""Retail Domain Module - Sales, inventory, and product analytics."""
from typing import Dict, Any, List
import pandas as pd
from .base import BaseDomain


def enrich(df):
    """Enrich dataframe with retail-specific features."""
    if "profit" in df.columns and "sales" in df.columns:
        df["margin"] = df["profit"] / df["sales"]
    return df


def domain_kpis(df):
    """Calculate retail-specific KPIs."""
    out = {}
    if "sales" in df.columns:
        out["Total Sales"] = df["sales"].sum()
    if "profit" in df.columns:
        out["Total Profit"] = df["profit"].sum()
    return out


class RetailDomain(BaseDomain):
    """Retail domain implementation for sales, inventory, and product analysis."""
    
    name = "retail"
    description = "Retail analytics: sales, inventory, product performance"
    required_columns = ["sales"]
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate if data contains retail-relevant columns."""
        return "sales" in df.columns or "revenue" in df.columns
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for retail analysis."""
        df = enrich(df)
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate retail KPIs."""
        return domain_kpis(df)
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate retail-specific insights."""
        insights = []
        if "Total Sales" in kpis:
            insights.append(f"Total Sales: ${kpis['Total Sales']:,.0f}")
        if "Total Profit" in kpis:
            insights.append(f"Total Profit: ${kpis['Total Profit']:,.0f}")
        return insights
