"""Data Profiling Engine for Sreejita v1.6

Provides statistical analysis and data quality profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataProfiler:
    """Enterprise-grade data profiling."""
    
    def __init__(self):
        self.profile_data: Dict[str, Any] = {}
    
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing profile data
        """
        self.profile_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_info": self._dataset_info(df),
            "column_profiles": self._column_profiles(df),
            "statistical_summary": self._statistical_summary(df),
            "missing_values": self._missing_analysis(df),
            "outliers": self._outlier_analysis(df)
        }
        return self.profile_data
    
    def _dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset-level information."""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    
    def _column_profiles(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Profile each column."""
        profiles = {}
        for col in df.columns:
            profiles[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percent": round(df[col].isnull().sum() / len(df) * 100, 2),
                "unique_count": int(df[col].nunique()),
                "unique_percent": round(df[col].nunique() / len(df) * 100, 2)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                profiles[col].update({
                    "mean": float(df[col].mean()) if df[col].notna().any() else None,
                    "std": float(df[col].std()) if df[col].notna().any() else None,
                    "min": float(df[col].min()) if df[col].notna().any() else None,
                    "max": float(df[col].max()) if df[col].notna().any() else None
                })
        
        return profiles
    
    def _statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summaries."""
        numeric_df = df.select_dtypes(include=[np.number])
        return {
            "numeric_columns": len(numeric_df.columns),
            "categorical_columns": len(df.select_dtypes(exclude=[np.number]).columns),
            "summary_stats": numeric_df.describe().to_dict() if len(numeric_df.columns) > 0 else {}
        }
    
    def _missing_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values."""
        missing = df.isnull().sum()
        return {
            "total_missing_cells": int(missing.sum()),
            "percent_missing": round(missing.sum() / (len(df) * len(df.columns)) * 100, 2),
            "columns_with_missing": missing[missing > 0].to_dict()
        }
    
    def _outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect outliers using IQR method."""
        outliers = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outlier_count = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
            if outlier_count > 0:
                outliers[col] = {
                    "count": int(outlier_count),
                    "percent": round(outlier_count / len(df) * 100, 2),
                    "bounds": {"lower": float(lower), "upper": float(upper)}
                }
        
        return outliers
    
    def get_report(self) -> Dict[str, Any]:
        """Get formatted profile report."""
        return self.profile_data
