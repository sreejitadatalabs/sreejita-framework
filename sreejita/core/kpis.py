from typing import Dict, Any, Optional
import pandas as pd


def compute_kpis(
    df: pd.DataFrame,
    base_kpis: Optional[Dict[str, Any]] = None,
    domain_kpis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge structural, base, and domain-specific KPIs into a single dictionary.

    This function is intentionally lightweight and deterministic.
    It does not calculate KPIs itself — it only consolidates them
    for reporting, insights, and narrative layers.

    Args:
        df: Input dataframe
        base_kpis: Framework-level KPIs (e.g., quality, volume)
        domain_kpis: Domain-specific KPIs (e.g., LOS, ROAS, GMROI)

    Returns:
        Dictionary of consolidated KPIs
    """

    # -----------------------------
    # Structural KPIs (always present)
    # -----------------------------
    kpis: Dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
    }

    # -----------------------------
    # Base KPIs (optional)
    # -----------------------------
    if isinstance(base_kpis, dict):
        for key, value in base_kpis.items():
            # Avoid accidental overwrite of structural KPIs
            if key not in kpis:
                kpis[key] = value

    # -----------------------------
    # Domain KPIs (optional)
    # -----------------------------
    if isinstance(domain_kpis, dict):
        for key, value in domain_kpis.items():
            kpis[key] = value

    return kpis
