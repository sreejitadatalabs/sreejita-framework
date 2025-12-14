"""
Sreejita Framework v1.7

Production-grade data analytics framework with
domain abstraction and automation support.
"""

from .__version__ import __version__

# Keep package init lightweight and safe
# Heavy modules (domains, ML, automation) should be imported explicitly by users



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

# Machine Learning Module
from .ml import PredictiveAnalytics, AutoML

__all__ = [
    "__version__",
    "clean_dataframe",
    "InsightGenerator",
    "compute_kpis",
    "BaseDomain",
    "RetailDomain",
    "EcommerceDomain",
    "CustomerDomain",
    "TextDomain",
    "FinanceDomain",
    "get_domain",
    "DOMAIN_REGISTRY",
        "PredictiveAnalytics",
    "AutoML",
]
