from sreejita.core.policy import PolicyDecision
from sreejita.policy.rules import (
    check_min_confidence,
    block_if_unknown_domain
)


class PolicyEngine:
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def evaluate(self, decision):
        reasons = []
        status = "allowed"

        # Rule 1: Unknown domain
        result = block_if_unknown_domain(decision)
        if result:
            return PolicyDecision(
                status="blocked",
                reasons=[result["reason"]],
                actions={}
            )

        # Rule 2: Confidence gate
        result = check_min_confidence(decision, self.min_confidence)
        if result:
            status = "allowed_with_warning"
            reasons.append(result["reason"])

        return PolicyDecision(
            status=status,
            reasons=reasons,
            actions={}
        )
