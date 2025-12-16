from sreejita.policy.engine import PolicyEngine
from sreejita.core.decision import DecisionExplanation


def test_low_confidence_triggers_warning():
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain="retail",
        confidence=0.4,
        alternatives=[],
        signals={},
        rules_applied=[]
    )

    engine = PolicyEngine(min_confidence=0.6)
    policy = engine.evaluate(decision)

    assert policy.status == "allowed_with_warning"


def test_unknown_domain_blocked():
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain="unknown",
        confidence=0.9,
        alternatives=[],
        signals={},
        rules_applied=[]
    )

    engine = PolicyEngine()
    policy = engine.evaluate(decision)

    assert policy.status == "blocked"
