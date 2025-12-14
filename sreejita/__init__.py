"""Sreejita Framework v1.2 - Data Analytics & Domain Modules

Universal framework for structured data analysis with pluggable domain modules.
"""

from .__version__ import __version__

# Core Engine
from .core.cleaner import clean_dataframe
from .core.insights import correlation_insights
from .core.kpis import compute_kpis

# Domain API
from .domains import (
    BaseDomain,
    RetailDomain,
    EcommerceDomain,
    CustomerDomain,
    TextDomain,
    FinanceDomain,
    get_domain,
    DOMAIN_REGISTRY,
)

__all__ = [
    "__version__",
    "clean_dataframe",
    "InsightGenerator",
    "KPICalculator",
    "BaseDomain",
    "RetailDomain",
    "EcommerceDomain",
    "CustomerDomain",
    "TextDomain",
    "FinanceDomain",
    "get_domain",
    "DOMAIN_REGISTRY",
]
