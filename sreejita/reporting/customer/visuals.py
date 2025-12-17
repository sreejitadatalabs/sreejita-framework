"""
Customer Domain Visuals
-----------------------
Executive-grade visuals for customer analytics.
"""

from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt

from sreejita.visuals.formatters import format_axis_human_readable


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def generate_visuals(
    df: pd.DataFrame,
    kpis: Dict[str, Dict[str, Any]] | None = None
) -> List[plt.Figure]:
    """
    Generate all customer visuals.

    Returns:
        List of matplotlib Figure objects
    """

    figures: List[plt.Figure] = []

    if "transaction_date" in df.columns:
        figures.append(_customer_trend(df))

    if "customer_id" in df.columns:
        figures.append(_customer_distribution(df))

    if {"customer_id", "revenue"}.issubset(df.columns):
        figures.append(_customer_value_distribution(df))

    return figures


# ---------------------------------------------------------------------
# Visual Implementations
# ---------------------------------------------------------------------

def _customer_trend(df: pd.DataFrame) -> plt.Figure:
    """
    Customer activity trend over time.
    """

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    trend = (
        df.groupby(pd.Grouper(key="transaction_date", freq="M"))["customer_id"]
        .nunique()
    )

    fig, ax = plt.subplots()
    trend.plot(ax=ax, marker="o")

    ax.set_title("Active Customers Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Active Customers")

    format_axis_human_readable(ax, axis="y")

    fig.tight_layout()
    return fig


def _customer_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Distribution of transactions per customer.
    """

    txn_counts = df.groupby("customer_id").size()

    fig, ax = plt.subplots()
    txn_counts.plot(kind="hist", bins=20, ax=ax)

    ax.set_title("Transactions per Customer Distribution")
    ax.set_xlabel("Number of Transactions")
    ax.set_ylabel("Customer Count")

    format_axis_human_readable(ax, axis="y")

    fig.tight_layout()
    return fig


def _customer_value_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Revenue contribution per customer.
    """

    customer_value = (
        df.groupby("customer_id")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()
    customer_value.plot(kind="bar", ax=ax)

    ax.set_title("Customer Revenue Contribution")
    ax.set_xlabel("Customers")
    ax.set_ylabel("Revenue")

    format_axis_human_readable(ax, axis="y")

    # Avoid overcrowding x-axis
    ax.set_xticks([])

    fig.tight_layout()
    return fig
