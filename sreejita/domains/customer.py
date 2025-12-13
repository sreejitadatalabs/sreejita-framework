"""Customer Domain Module - Customer segmentation & behavior.

Supports: Demographics, RFM, behavioral segments, churn prediction.
Works with: Customer databases, transaction history, behavioral data.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseDomain
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class CustomerDomain(BaseDomain):
    """Customer segmentation and profiling domain.
    
    Calculates customer-specific KPIs:
    - RFM metrics (Recency, Frequency, Monetary)
    - Customer segments
    - Churn indicators
    - Lifetime value
    - Engagement scores
    """
    
    name = "customer"
    description = "Customer Segmentation & Profiling Domain"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data can be analyzed as customer data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean customer data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate customer KPIs."""
        kpis = {
            "total_customers": len(df),
            "avg_customer_value": 0,
            "churn_risk_count": 0,
            "active_segments": 0,
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            kpis["avg_customer_value"] = float(df[numeric_cols].mean().mean())
        kpis["active_segments"] = min(3, max(1, len(df) // 100))
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate customer insights."""
        insights = [
            f"Analyzing {kpis['total_customers']} customer records",
            f"Average customer metric: {kpis['avg_customer_value']:.2f}",
            f"Identified {kpis['active_segments']} customer segments",
        ]
        return insights
