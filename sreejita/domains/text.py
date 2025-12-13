"""Text/NLP Domain Module - Adapter for text analytics.

Purpose: Analyze text data by converting it to structured features.
Supports: Sentiment, topics, word frequencies, embeddings.
Works with: Reviews, comments, feedback, social media.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseDomain


class TextDomain(BaseDomain):
    """Text/NLP domain module.
    
    Analyzes text data extracted from reviews, comments, etc.
    Expected inputs: sentiment_score, topic, word_count, polarity, etc.
    
    NOTE: This expects preprocessed text features.
    Use external NLP tool (spaCy, transformers) to extract features first.
    """
    
    name = "text"
    description = "Text/NLP Analytics Domain"
    required_columns = []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check if data can be analyzed as text features."""
        return len(df) > 0 and isinstance(df, pd.DataFrame)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text features."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Neutral')
        return df
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate text feature KPIs."""
        kpis = {
            "total_texts": len(df),
            "avg_sentiment": 0,
            "positive_ratio": 0,
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            kpis["avg_sentiment"] = float(df[numeric_cols].mean().mean())
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate text insights."""
        insights = [
            f"Analyzed {kpis['total_texts']} text records",
            f"Average sentiment score: {kpis['avg_sentiment']:.3f}",
        ]
        return insights
