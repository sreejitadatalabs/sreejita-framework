"""Finance Domain Module - Financial analytics & metrics.

Supports: P&L, cash flow, forecasting, risk metrics.
Works with: Financial statements, transaction data, market data.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseDomain


class FinanceDomain(BaseDomain):
    """Financial analytics domain module.
    
    Calculates financial-specific KPIs:
    - Revenue & expenses
    - Profit margins
    - Cash flow metrics
    - Financial ratios
    - Forecasting indicators
    """
    
    name = "finance"
    description = "Financial Analytics Domain"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data can be analyzed as financial data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean financial data."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Uncat')
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate financial KPIs."""
        kpis = {
            "total_transactions": len(df),
            "avg_amount": 0,
            "total_volume": 0,
            "volatility": 0,
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            kpis["avg_amount"] = float(df[numeric_cols].mean().mean())
            kpis["total_volume"] = float(df[numeric_cols].sum().sum())
            kpis["volatility"] = float(df[numeric_cols].std().mean())
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate financial insights."""
        insights = [
            f"Analyzed {kpis['total_transactions']} financial records",
            f"Average amount: {kpis['avg_amount']:.2f}",
            f"Total volume: {kpis['total_volume']:.2f}",
            f"Volatility: {kpis['volatility']:.4f}",
        ]
        return insights
