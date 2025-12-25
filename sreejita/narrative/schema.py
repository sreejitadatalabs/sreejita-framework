from typing import List, Dict, Any
from dataclasses import dataclass, field


# =====================================================
# v3.5 NARRATIVE INPUT SCHEMA
# =====================================================
# This schema defines the ONLY data that an LLM is
# allowed to see. Do not extend casually.
# =====================================================


ALLOWED_CONFIDENCE_BANDS = {"HIGH", "MEDIUM", "LOW"}
ALLOWED_LEVELS = {"RISK", "WARNING", "INFO"}


@dataclass(frozen=True)
class NarrativeInsight:
    """
    A single insight already approved by the system.
    LLM may ONLY rewrite this, never invent new ones.
    """
    level: str
    title: str
    description: str

    def __post_init__(self):
        if self.level not in ALLOWED_LEVELS:
            raise ValueError(f"Invalid insight level: {self.level}")

        if not self.title or not isinstance(self.title, str):
            raise ValueError("Insight title must be a non-empty string")

        if not self.description or not isinstance(self.description, str):
            raise ValueError("Insight description must be a non-empty string")


@dataclass(frozen=True)
class NarrativeAction:
    """
    A recommended action already approved by the system.
    LLM may ONLY describe this action, never modify it.
    """
    action: str
    priority: str
    timeline: str

    def __post_init__(self):
        if not self.action or not isinstance(self.action, str):
            raise ValueError("Action must be a non-empty string")

        if not self.priority or not isinstance(self.priority, str):
            raise ValueError("Priority must be a non-empty string")

        if not self.timeline or not isinstance(self.timeline, str):
            raise ValueError("Timeline must be a non-empty string")


@dataclass(frozen=True)
class NarrativeInput:
    """
    Canonical input contract for the v3.5 LLM narrative layer.

    WARNING:
    - Do NOT add KPIs
    - Do NOT add numbers
    - Do NOT add historical comparisons
    - Do NOT add rules or logic
    """

    run_id: str
    domain: str
    summary_level: str = "EXECUTIVE"

    insights: List[NarrativeInsight] = field(default_factory=list)
    actions: List[NarrativeAction] = field(default_factory=list)

    confidence_band: str = "MEDIUM"

    constraints: Dict[str, bool] = field(
        default_factory=lambda: {
            "no_new_insights": True,
            "no_new_actions": True,
            "no_numbers": True,
        }
    )

    def __post_init__(self):
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string")

        if not self.domain or not isinstance(self.domain, str):
            raise ValueError("domain must be a non-empty string")

        if self.confidence_band not in ALLOWED_CONFIDENCE_BANDS:
            raise ValueError(
                f"Invalid confidence_band: {self.confidence_band}. "
                f"Allowed: {ALLOWED_CONFIDENCE_BANDS}"
            )

        if not isinstance(self.insights, list):
            raise ValueError("insights must be a list")

        if not isinstance(self.actions, list):
            raise ValueError("actions must be a list")

        if not isinstance(self.constraints, dict):
            raise ValueError("constraints must be a dictionary")

        # HARD SAFETY CHECKS
        if self.constraints.get("no_new_insights") is not True:
            raise ValueError("no_new_insights constraint must be True")

        if self.constraints.get("no_new_actions") is not True:
            raise ValueError("no_new_actions constraint must be True")

        if self.constraints.get("no_numbers") is not True:
            raise ValueError("no_numbers constraint must be True")

    # -------------------------------------------------
    # SERIALIZATION (Safe for prompts)
    # -------------------------------------------------

    def to_prompt_dict(self) -> Dict[str, Any]:
        """
        Convert to a strictly controlled dictionary
        that is safe to pass into an LLM prompt.
        """

        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "summary_level": self.summary_level,
            "confidence_band": self.confidence_band,
            "insights": [
                {
                    "level": i.level,
                    "title": i.title,
                    "description": i.description,
                }
                for i in self.insights
            ],
            "actions": [
                {
                    "action": a.action,
                    "priority": a.priority,
                    "timeline": a.timeline,
                }
                for a in self.actions
            ],
            "constraints": self.constraints,
        }
