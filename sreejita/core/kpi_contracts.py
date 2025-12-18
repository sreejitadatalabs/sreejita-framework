from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class KPIContract:
    name: str
    unit: Literal["currency", "percent", "ratio", "count", "raw"]
    direction: Literal["higher_is_better", "lower_is_better", "neutral"]
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
