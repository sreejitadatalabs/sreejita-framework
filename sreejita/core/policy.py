from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PolicyDecision:
    status: str  # allowed | allowed_with_warning | blocked
    reasons: List[str]
    actions: Dict[str, Any]
