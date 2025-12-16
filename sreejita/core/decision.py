from dataclasses import dataclass, field
from typing import Any, Dict, List
from datetime import datetime


@dataclass
class DecisionExplanation:
    decision_type: str
    selected_domain: str
    confidence: float

    alternatives: List[Dict[str, Any]]
    signals: Dict[str, Any]
    rules_applied: List[str]

    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
