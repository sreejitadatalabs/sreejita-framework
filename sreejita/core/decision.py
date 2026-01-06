# =====================================================
# DECISION EXPLANATION — UNIVERSAL (FINAL, HARDENED)
# Sreejita Framework v3.6
# =====================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class DecisionExplanation:
    """
    Canonical decision explanation object.

    Used by:
    - Router
    - Orchestrator
    - Streamlit UI
    - Observability
    - Debugging & audit

    GUARANTEES:
    - Always serializable
    - Always engine-attached
    - Always explainable
    """

    # -------------------------------------------------
    # CORE DECISION
    # -------------------------------------------------
    decision_type: str
    selected_domain: str
    confidence: float

    # -------------------------------------------------
    # EXPLAINABILITY
    # -------------------------------------------------
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    rules_applied: List[str] = field(default_factory=list)

    # -------------------------------------------------
    # SCORING & TRACEABILITY
    # -------------------------------------------------
    domain_scores: Optional[Dict[str, Any]] = None
    fingerprint: Optional[str] = None

    # -------------------------------------------------
    # EXECUTION CONTEXT (ATTACHED LATER)
    # -------------------------------------------------
    engine: Any = field(default=None, repr=False)

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------
    decision_id: str = field(
        default_factory=lambda: f"DEC-{uuid.uuid4().hex[:10]}"
    )

    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # =================================================
    # SAFETY & NORMALIZATION
    # =================================================
    def __post_init__(self):
        # Confidence must be numeric and bounded
        try:
            self.confidence = float(self.confidence)
        except Exception:
            self.confidence = 0.0

        self.confidence = max(0.0, min(self.confidence, 1.0))

        # Normalize containers
        if self.alternatives is None:
            self.alternatives = []

        if self.signals is None:
            self.signals = {}

        if self.rules_applied is None:
            self.rules_applied = []

    # =================================================
    # SERIALIZATION (STREAMLIT / JSON SAFE)
    # =================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert decision into a JSON-safe dictionary.
        Engine is intentionally excluded.
        """
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "selected_domain": self.selected_domain,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "signals": self.signals,
            "rules_applied": self.rules_applied,
            "domain_scores": self.domain_scores,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
        }

    # =================================================
    # DEBUG FRIENDLY STRING
    # =================================================
    def __str__(self) -> str:
        return (
            f"[Decision {self.decision_id}] "
            f"{self.decision_type} → {self.selected_domain} "
            f"(confidence={self.confidence:.2f})"
        )
