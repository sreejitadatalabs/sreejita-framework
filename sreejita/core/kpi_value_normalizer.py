from .kpi_normalizer import KPI_REGISTRY


def normalize_kpi_value(kpi_name: str, value):
    """
    Ensures KPI values respect their declared unit.
    """
    contract = KPI_REGISTRY.get(kpi_name)

    if not contract or value is None:
        return value

    # Percent KPIs must be 0–100
    if contract.unit == "percent":
        if 0 <= value <= 1:
            return value * 100
        return value

    return value
