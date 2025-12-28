
# sreejita/narrative/schema.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExecutiveFinding:
    title: str
    explanation: str
    impact: Optional[str] = None


@dataclass
class ActionItem:
    action: str
    owner: str
    timeline: str
    success_kpi: str


@dataclass
class NarrativeOutput:
    executive_summary: List[str]
    key_findings: List[ExecutiveFinding]
    financial_impact: List[str]
    risks: List[str]
    action_plan: List[ActionItem]
