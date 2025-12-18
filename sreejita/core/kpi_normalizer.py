from typing import Dict
from .kpi_contracts import KPIContract


KPI_REGISTRY: Dict[str, KPIContract] = {

    # ---------- Retail / Commerce ----------
    "discount_rate": KPIContract(
        name="discount_rate",
        unit="percent",
        direction="lower_is_better",
        description="Average discount applied on transactions",
        min_value=0,
        max_value=100,
    ),

    "profit_margin": KPIContract(
        name="profit_margin",
        unit="percent",
        direction="higher_is_better",
        description="Profit as a percentage of revenue",
        min_value=0,
        max_value=100,
    ),

    "total_sales": KPIContract(
        name="total_sales",
        unit="currency",
        direction="higher_is_better",
        description="Total revenue generated",
    ),

    # ---------- Finance ----------
    "net_profit": KPIContract(
        name="net_profit",
        unit="currency",
        direction="higher_is_better",
        description="Net profit after costs",
    ),

    # ---------- Ops ----------
    "avg_cycle_time": KPIContract(
        name="avg_cycle_time",
        unit="raw",
        direction="lower_is_better",
        description="Average time to complete an operation",
    ),

    "on_time_rate": KPIContract(
        name="on_time_rate",
        unit="percent",
        direction="higher_is_better",
        description="Percentage of operations completed on time",
        min_value=0,
        max_value=100,
    ),

    # ---------- Healthcare ----------
    "avg_outcome_score": KPIContract(
        name="avg_outcome_score",
        unit="raw",
        direction="higher_is_better",
        description="Average patient outcome score",
    ),

    # ---------- Marketing ----------
    "conversion_rate": KPIContract(
        name="conversion_rate",
        unit="percent",
        direction="higher_is_better",
        description="Percentage of users converted",
        min_value=0,
        max_value=100,
    ),
}
