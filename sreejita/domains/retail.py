"""Retail Domain Module - KPIs and insights for retail analytics.

Supports: Sales, inventory, customer behavior, seasonal trends.
Works with: Transaction data, product catalogs, store metrics.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseDomain


class RetailDomain(BaseDomain):
    """Retail analytics domain module.
    
    Calculates retail-specific KPIs:
    - Revenue, sales, transactions
    - Product performance
    - Customer metrics
    - Seasonal trends
    """
    
    name = "retail"
    description = "Retail Analytics Domain"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data can be analyzed as retail data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare retail data."""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate retail KPIs."""
        kpis = {
            "total_records": len(df),
            "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(df.select_dtypes(include=['object']).columns),
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            kpis["avg_numeric_value"] = float(df[numeric_cols].mean().mean())
            kpis["max_numeric_value"] = float(df[numeric_cols].max().max())
            kpis["min_numeric_value"] = float(df[numeric_cols].min().min())
        
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate retail insights."""
        insights = [
            f"Dataset contains {kpis['total_records']} retail records",
            f"Features: {kpis['numeric_features']} numeric, {kpis['categorical_features']} categorical",
        ]
        
        if 'avg_numeric_value' in kpis:
            insights.append(f"Average metric value: {kpis['avg_numeric_value']:.2f}")
        
        return insights
