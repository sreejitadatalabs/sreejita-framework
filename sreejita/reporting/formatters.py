from typing import Optional

def fmt_currency(value: Optional[float]) -> str:
    """
    Canonical currency formatter for ALL reports.
    """
    if value is None:
        return "-"

    try:
        value = float(value)
    except Exception:
        return "-"

    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"


def fmt_percent(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{value*100:.{decimals}f}%"
    except Exception:
        return "-"
