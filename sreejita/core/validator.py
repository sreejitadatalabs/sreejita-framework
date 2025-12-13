"""Data Quality Validation Framework for Sreejita v1.6

Provides comprehensive data quality checks before processing.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Stores validation check result."""
    check_name: str
    passed: bool
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DataQualityValidator:
    """Enterprise-grade data quality validation."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
        self.passed = True
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[ValidationResult]]:
        self.results = []
        self.passed = True
        
        self._check_not_empty(df)
        self._check_no_null_columns(df)
        self._check_numeric_columns(df)
        self._check_duplicate_rows(df)
        self._check_column_names(df)
        self._check_memory_usage(df)
        
        if self.strict_mode:
            self.passed = all(r.passed for r in self.results)
        else:
            self.passed = all(r.passed for r in self.results if r.severity == 'error')
        
        return self.passed, self.results
    
    def _check_not_empty(self, df: pd.DataFrame):
        passed = len(df) > 0
        result = ValidationResult(
            check_name="empty_dataframe",
            passed=passed,
            severity="error",
            message=f"DataFrame has {len(df)} rows",
            details={"rows": len(df), "columns": len(df.columns)}
        )
        self.results.append(result)
    
    def _check_no_null_columns(self, df: pd.DataFrame):
        null_cols = df.columns[df.isnull().all()].tolist()
        passed = len(null_cols) == 0
        result = ValidationResult(
            check_name="null_columns",
            passed=passed,
            severity="error" if null_cols else "info",
            message=f"Found {len(null_cols)} entirely null columns",
            details={"null_columns": null_cols}
        )
        self.results.append(result)
    
    def _check_numeric_columns(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['number']).columns
        issues = {}
        for col in numeric_cols:
            inf_count = (df[col] == float('inf')).sum()
            if inf_count > 0:
                issues[col] = int(inf_count)
        
        passed = len(issues) == 0
        result = ValidationResult(
            check_name="numeric_validity",
            passed=passed,
            severity="warning" if issues else "info",
            message=f"Found {sum(issues.values())} invalid numeric values",
            details={"columns_with_inf": issues}
        )
        self.results.append(result)
    
    def _check_duplicate_rows(self, df: pd.DataFrame):
        dup_count = df.duplicated().sum()
        passed = dup_count == 0
        result = ValidationResult(
            check_name="duplicate_rows",
            passed=passed,
            severity="warning" if dup_count > 0 else "info",
            message=f"Found {dup_count} duplicate rows",
            details={"duplicates": int(dup_count)}
        )
        self.results.append(result)
    
    def _check_column_names(self, df: pd.DataFrame):
        invalid_cols = []
        for col in df.columns:
            if not isinstance(col, str) or col.strip() == "":
                invalid_cols.append(str(col))
        
        passed = len(invalid_cols) == 0
        result = ValidationResult(
            check_name="column_names",
            passed=passed,
            severity="error" if invalid_cols else "info",
            message=f"Found {len(invalid_cols)} invalid column names",
            details={"invalid_columns": invalid_cols}
        )
        self.results.append(result)
    
    def _check_memory_usage(self, df: pd.DataFrame):
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        passed = memory_mb < 1000
        result = ValidationResult(
            check_name="memory_usage",
            passed=passed or not self.strict_mode,
            severity="warning" if memory_mb > 1000 else "info",
            message=f"Memory usage: {memory_mb:.2f} MB",
            details={"memory_mb": round(memory_mb, 2)}
        )
        self.results.append(result)
    
    def get_report(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "failed_checks": sum(1 for r in self.results if not r.passed),
            "results": [{
                "check": r.check_name,
                "passed": r.passed,
                "severity": r.severity,
                "message": r.message,
                "details": r.details
            } for r in self.results]
        }
