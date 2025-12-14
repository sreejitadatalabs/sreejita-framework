"""Core Engine Module - Data cleaning, insights generation, and KPI computation."""

from .cleaner import clean_dataframe
from .insights import correlation_insights
from .kpis import compute_kpis

__all__ = [
    "clean_dataframe",
    "correlation_insights",
    "compute_kpis",
]
