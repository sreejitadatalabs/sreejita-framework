from sreejita.core.decision import DomainDecision
from sreejita.domains.retail import RetailDomainDetector
from sreejita.domains.finance import FinanceDomainDetector
# add other domains here as you extend
# from sreejita.domains.healthcare import HealthcareDomainDetector
# from sreejita.domains.ops import OpsDomainDetector


DETECTORS = [
    RetailDomainDetector(),
    FinanceDomainDetector(),
    # add more detectors here
]


def decide_domain(df) -> DomainDecision:
    """
    Runs all domain detectors and selects the best domain
    using signal-weighted logic.
    """

    results = {}

    for detector in DETECTORS:
        result = detector.detect(df)
        results[detector.domain] = result

    # ----------------------------------
    # Prefer domains with PRIMARY signals
    # ----------------------------------
    primary_domains = {
        d: r for d, r in results.items()
        if r.signals.get("primary")
    }

    candidates = primary_domains if primary_domains else results

    selected_domain, best_result = max(
        candidates.items(),
        key=lambda x: x[1].confidence
    )

    return DomainDecision(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=best_result.confidence,
        rules_applied=[
            "signal_weighted_detection",
            "primary_signal_preference" if primary_domains else "fallback_selection",
        ],
        signals={
            d: {
                "primary": list(r.signals["primary"]),
                "secondary": list(r.signals["secondary"]),
                "generic": list(r.signals["generic"]),
            }
            for d, r in results.items()
        },
    )