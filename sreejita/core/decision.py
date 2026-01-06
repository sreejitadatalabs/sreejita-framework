# =====================================================
# DECISION EXPLANATION — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6 STABILIZED
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
    - Domain Router
    - Orchestrator
    - Streamlit UI
    - Observability / Audit
    - Debugging

    HARD GUARANTEES:
    - Always serializable
    - Never crashes on missing fields
    - Engine is attached lazily & safely
    - Confidence is bounded
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
    # EXECUTION CONTEXT (NEVER SERIALIZED)
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
        # -----------------------------
        # Confidence normalization
        # -----------------------------
        try:
            self.confidence = float(self.confidence)
        except Exception:
            self.confidence = 0.0

        self.confidence = max(0.0, min(self.confidence, 1.0))

        # -----------------------------
        # Container normalization
        # -----------------------------
        if not isinstance(self.alternatives, list):
            self.alternatives = []

        if not isinstance(self.signals, dict):
            self.signals = {}

        if not isinstance(self.rules_applied, list):
            self.rules_applied = []

        if self.domain_scores is not None and not isinstance(self.domain_scores, dict):
            self.domain_scores = None

        # -----------------------------
        # Core field safety
        # -----------------------------
        if not isinstance(self.decision_type, str):
            self.decision_type = "unknown_decision"

        if not isinstance(self.selected_domain, str):
            self.selected_domain = "generic"

    # =================================================
    # SERIALIZATION (STREAMLIT / JSON SAFE)
    # =================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert decision into a JSON-safe dictionary.

        NOTE:
        - `engine` is intentionally excluded
        - All values are UI-safe
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
    # SAFE ENGINE ATTACHMENT
    # =================================================
    def attach_engine(self, engine: Any) -> None:
        """
        Attach domain execution engine safely.
        """
        self.engine = engine

    # =================================================
    # DEBUG / LOGGING
    # =================================================
    def __str__(self) -> str:
        return (
            f"[Decision {self.decision_id}] "
            f"{self.decision_type} → {self.selected_domain} "
            f"(confidence={self.confidence:.2f})"
        )

    __repr__ = __str__
