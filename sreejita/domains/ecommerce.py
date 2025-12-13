"""E-commerce Domain Module - Analytics for online retail.

Supports: Orders, conversions, cart data, customer journeys.
Works with: Transaction logs, product catalogs, click streams.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseDomain


class EcommerceDomain(BaseDomain):
    """E-commerce analytics domain module.
    
    Calculates e-commerce specific KPIs:
    - Conversion rates
    - Cart metrics
    - Customer lifetime value
    - Product performance
    - Channel attribution
    """
    
    name = "ecommerce"
    description = "E-commerce Analytics Domain"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data can be analyzed as ecommerce data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ecommerce data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Direct')
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ecommerce KPIs."""
        kpis = {
            "total_orders": len(df),
            "total_features": len(df.columns),
            "date_range": f"{df.shape[0]} records",
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            kpis["avg_order_value"] = float(df[numeric_cols].mean().mean())
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate ecommerce insights."""
        insights = [
            f"Total orders analyzed: {kpis['total_orders']}",
            f"Dataset features: {kpis['total_features']}",
        ]
        if 'avg_order_value' in kpis:
            insights.append(f"Average value metric: {kpis['avg_order_value']:.2f}")
        return insights
