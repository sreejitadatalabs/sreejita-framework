"""Base Domain Class - Foundation for all domain modules.

The BaseDomain class provides the interface that all specific domains
(Retail, E-commerce, Customer, Text, Finance) must implement.

Design Pattern: Strategy Pattern + Plugin Architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd


class BaseDomain(ABC):
    """Abstract base class for all domain modules.
    
    All domain modules must inherit from this class and implement
    the abstract methods to ensure consistent behavior.
    """
    
    name: str = "generic"
    description: str = "Generic domain"
    required_columns: List[str] = []
    
    def __init__(self):
        """Initialize domain module."""
        self.kpis = {}
        self.insights = []
        
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate if DataFrame is compatible with this domain."""
        pass
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Domain-specific preprocessing."""
        pass
    
    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate domain-specific KPIs."""
        pass
    
    @abstractmethod
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Generate domain-specific insights."""
        pass
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run full domain analysis pipeline."""
        if not self.validate_data(df):
            raise ValueError(f"Data validation failed for {self.name}")
        
        processed_df = self.preprocess(df)
        kpis = self.calculate_kpis(processed_df)
        insights = self.generate_insights(processed_df, kpis)
        
        return {
            "domain": self.name,
            "description": self.description,
            "data": processed_df,
            "kpis": kpis,
            "insights": insights,
        }
