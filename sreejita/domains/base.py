from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd


class BaseDomain(ABC):
    """
    Abstract base class for all domain modules.
    Strategy Pattern + Plugin Architecture
    """

    name: str = "generic"
    description: str = "Generic domain"
    required_columns: List[str] = []

    def __init__(self):
        self.kpis = {}
        self.insights = []

    # --------------------------------------------------
    # OPTIONAL VALIDATION (DEFAULT SAFE)
    # --------------------------------------------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Optional domain-level validation.
        Default = always valid.

        Domains may override this if strict schema
        validation is required.
        """
        return True

    # --------------------------------------------------
    # REQUIRED DOMAIN CONTRACTS
    # --------------------------------------------------

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Any]:
        pass

    # --------------------------------------------------
    # PIPELINE
    # --------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
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
