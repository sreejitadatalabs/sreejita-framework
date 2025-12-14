"""Customer Domain Module - Segmentation and behavioral analytics."""
from typing import Dict, List, Any
import pandas as pd
from .base import BaseDomain


class CustomerDomain(BaseDomain):
    """Customer analytics domain module."""
    
    name = "customer"
    description = "Customer Analytics: segmentation, RFM, churn prediction"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate customer data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess customer data."""
        df = df.copy()
        df = df.fillna(0)
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate customer KPIs."""
        kpis = {}
        kpis["Total Customers"] = len(df)
        if "revenue" in df.columns:
            kpis["Average Customer Value"] = df["revenue"].sum() / len(df) if len(df) > 0 else 0
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate customer insights."""
        insights = []
        if "Total Customers" in kpis:
            insights.append(f"Total Customers: {kpis['Total Customers']}")
        if "Average Customer Value" in kpis:
            insights.append(f"Avg Customer Value: ${kpis['Average Customer Value']:.2f}")
        return insights 
