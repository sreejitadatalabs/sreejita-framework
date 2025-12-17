"""
Customer Domain KPIs
--------------------
Customer-specific KPI calculations.
Fully self-contained and import-safe.
"""

from typing import Dict, Any
import pandas as pd


# ---------------------------------------------------------------------
# Local Helpers
# ---------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator in (0, None):
        return None
    return numerator / denominator


def _percentage(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100


def _validate_columns(df: pd.DataFrame, required: list[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _format(value: Any, unit: str | None = None, currency: bool = False) -> str:
    if value is None:
        return "N/A"
    if currency:
        return f"${value:,.2f}"
    if unit == "%":
        return f"{value:.1f}%"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


# ---------------------------------------------------------------------
# KPI Computation
# ---------------------------------------------------------------------

def compute_customer_kpis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    _validate_columns(
        df,
        ["customer_id", "transaction_date", "revenue"]
    )

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    total_customers = df["customer_id"].nunique()
    active_customers = df[df["revenue"] > 0]["customer_id"].nunique()

    repeat_customers = (
        df.groupby("customer_id")
        .size()
        .reset_index(name="txn_count")
        .query("txn_count > 1")["customer_id"]
        .nunique()
    )

    churn_rate = _compute_churn_rate(df)
    retention_rate = _percentage(1 - churn_rate) if churn_rate is not None else None

    avg_customer_value = _safe_divide(
        df["revenue"].sum(),
        total_customers
    )

    purchase_frequency = _safe_divide(len(df), total_customers)

    return {
        "total_customers": _kpi("Total Customers", total_customers),
        "active_customers": _kpi("Active Customers", active_customers),
        "repeat_customers": _kpi("Repeat Customers", repeat_customers),
        "churn_rate": _kpi("Churn Rate", _percentage(churn_rate), unit="%"),
        "retention_rate": _kpi("Retention Rate", retention_rate, unit="%"),
        "average_customer_value": _kpi(
            "Avg. Customer Value", avg_customer_value, currency=True
        ),
        "purchase_frequency": _kpi(
            "Purchase Frequency", purchase_frequency
        ),
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _compute_churn_rate(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None

    last_date = df["transaction_date"].max()
    cutoff = last_date - pd.Timedelta(days=90)

    last_activity = (
        df.groupby("customer_id")["transaction_date"]
        .max()
        .reset_index()
    )

    churned = last_activity[
        last_activity["transaction_date"] < cutoff
    ]["customer_id"].nunique()

    total = last_activity["customer_id"].nunique()

    return _safe_divide(churned, total)


def _kpi(
    label: str,
    value: Any,
    unit: str | None = None,
    currency: bool = False
) -> Dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "formatted": _format(value, unit=unit, currency=currency),
        "unit": unit,
    }
