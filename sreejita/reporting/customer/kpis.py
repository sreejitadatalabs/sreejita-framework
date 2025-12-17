"""
Customer Domain KPIs
--------------------
This module defines all customer-related KPIs used in reports.
Retail is treated as a reference pattern, not a dependency.
"""

from typing import Dict, Any
import pandas as pd

from sreejita.core.kpis import (
    safe_divide,
    percentage,
    average,
)
from sreejita.core.validator import require_columns
from sreejita.reporting.formatter import format_kpi_value


# ---------------------------------------------------------------------
# Input Schema Contract
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "customer_id",
    "transaction_date",
    "revenue",
]

OPTIONAL_COLUMNS = [
    "is_new_customer",
]


# ---------------------------------------------------------------------
# KPI Computation
# ---------------------------------------------------------------------

def compute_customer_kpis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute all Customer KPIs.

    Returns:
        Dict[str, Dict[str, Any]]
        {
            "total_customers": {...},
            "active_customers": {...},
            ...
        }
    """

    require_columns(df, REQUIRED_COLUMNS)

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    total_customers = df["customer_id"].nunique()

    active_customers = (
        df[df["revenue"] > 0]["customer_id"].nunique()
    )

    repeat_customers = (
        df.groupby("customer_id")
        .size()
        .reset_index(name="txn_count")
        .query("txn_count > 1")["customer_id"]
        .nunique()
    )

    new_customers = (
        df[df.get("is_new_customer", False) == True]["customer_id"].nunique()
        if "is_new_customer" in df.columns
        else None
    )

    churn_rate = _compute_churn_rate(df)
    retention_rate = percentage(1 - churn_rate) if churn_rate is not None else None

    avg_customer_value = safe_divide(
        df["revenue"].sum(),
        total_customers
    )

    purchase_frequency = safe_divide(
        len(df),
        total_customers
    )

    # -----------------------------------------------------------------
    # Normalized KPI Output (Retail Parity)
    # -----------------------------------------------------------------

    return {
        "total_customers": _kpi(
            value=total_customers,
            label="Total Customers"
        ),
        "active_customers": _kpi(
            value=active_customers,
            label="Active Customers"
        ),
        "repeat_customers": _kpi(
            value=repeat_customers,
            label="Repeat Customers"
        ),
        "new_customers": _kpi(
            value=new_customers,
            label="New Customers"
        ),
        "churn_rate": _kpi(
            value=percentage(churn_rate) if churn_rate is not None else None,
            label="Churn Rate",
            unit="%"
        ),
        "retention_rate": _kpi(
            value=retention_rate,
            label="Retention Rate",
            unit="%"
        ),
        "average_customer_value": _kpi(
            value=avg_customer_value,
            label="Avg. Customer Value",
            currency=True
        ),
        "purchase_frequency": _kpi(
            value=purchase_frequency,
            label="Purchase Frequency"
        ),
    }


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def _compute_churn_rate(df: pd.DataFrame) -> float | None:
    """
    Compute churn rate based on customer inactivity.

    Heuristic:
    - Customer is churned if no activity in last 90 days
    """

    if df.empty:
        return None

    last_date = df["transaction_date"].max()
    cutoff = last_date - pd.Timedelta(days=90)

    last_activity = (
        df.groupby("customer_id")["transaction_date"]
        .max()
        .reset_index()
    )

    churned_customers = last_activity[
        last_activity["transaction_date"] < cutoff
    ]["customer_id"].nunique()

    total_customers = last_activity["customer_id"].nunique()

    return safe_divide(churned_customers, total_customers)


def _kpi(
    value: Any,
    label: str,
    unit: str | None = None,
    currency: bool = False
) -> Dict[str, Any]:
    """
    Standard KPI envelope to ensure report compatibility.
    """

    return {
        "label": label,
        "value": value,
        "formatted": format_kpi_value(
            value,
            unit=unit,
            currency=currency
        ),
        "unit": unit,
    }
