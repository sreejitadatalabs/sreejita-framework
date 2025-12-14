"""E-commerce Domain Module - Analytics for online retail."""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseDomain


class EcommerceDomain(BaseDomain):
    """E-commerce analytics domain module."""
    
    name = "ecommerce"
    description = "E-commerce Analytics: conversions, cart metrics, CLV"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data can be analyzed as ecommerce data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ecommerce data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate e-commerce KPIs."""
        kpis = {}
        if "revenue" in df.columns:
            kpis["Total Revenue"] = df["revenue"].sum()
        if "transactions" in df.columns:
            kpis["Total Transactions"] = len(df)
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate e-commerce specific insights."""
        insights = []
        if "Total Revenue" in kpis:
            insights.append(f"Total Revenue: ${kpis['Total Revenue']:,.0f}")
        if "Total Transactions" in kpis:
            insights.append(f"Total Transactions: {kpis['Total Transactions']}")
        return insights
        
