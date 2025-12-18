from typing import Dict, Any

from sreejita.core.kpi_value_normalizer import normalize_kpi_value
from sreejita.core.kpi_normalizer import KPI_REGISTRY


def normalize_kpis(raw_kpis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize KPI values based on KPI contracts.
    Ensures consistent units (e.g., percent vs fraction).
    """
    normalized = {}

    for kpi_name, value in raw_kpis.items():
        if kpi_name in KPI_REGISTRY:
            normalized[kpi_name] = normalize_kpi_value(kpi_name, value)
        else:
            # Unknown KPIs pass through unchanged
            normalized[kpi_name] = value

    return normalized
