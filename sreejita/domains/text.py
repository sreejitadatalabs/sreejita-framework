"""Text Domain Module - NLP and sentiment analysis."""
from typing import Dict, List, Any
import pandas as pd
from .base import BaseDomain


class TextDomain(BaseDomain):
    """Text/NLP analytics domain module."""
    
    name = "text"
    description = "Text Analytics: sentiment, topics, NLP features"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate text data."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text data."""
        df = df.copy()
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate text KPIs."""
        kpis = {}
        kpis["Total Records"] = len(df)
        if "sentiment" in df.columns:
            kpis["Avg Sentiment"] = df["sentiment"].mean()
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate text insights."""
        insights = []
        if "Total Records" in kpis:
            insights.append(f"Total Records Analyzed: {kpis['Total Records']}")
        if "Avg Sentiment" in kpis:
            insights.append(f"Average Sentiment: {kpis['Avg Sentiment']:.2f}")
        return insights
