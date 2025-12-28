# sreejita/narrative/engine.py
from typing import Dict
from .schema import NarrativeOutput
from .rules import healthcare_rules
from .formatter import executive_summary


def build_narrative(
    domain: str,
    kpis: Dict,
    insights: list,
    recommendations: list,
) -> NarrativeOutput:

    if domain == "healthcare":
        result = healthcare_rules(kpis)
    else:
        result = {"findings": [], "financial": [], "risks": [], "actions": []}

    summary = executive_summary(result["findings"])

    return NarrativeOutput(
        executive_summary=summary,
        key_findings=result["findings"],
        financial_impact=result["financial"],
        risks=result["risks"],
        action_plan=result["actions"],
    )
