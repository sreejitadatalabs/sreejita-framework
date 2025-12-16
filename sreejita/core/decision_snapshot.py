from sreejita.core.decision import DecisionExplanation

def decision_to_dict(decision: DecisionExplanation) -> dict:
    return {
        "decision_type": decision.decision_type,
        "selected_domain": decision.selected_domain,
        "confidence": decision.confidence,
        "alternatives": decision.alternatives,
        "signals": decision.signals,
        "rules_applied": decision.rules_applied,
        "timestamp": decision.timestamp,
    }
