from sreejita.domains.registry import registry
from sreejita.domains.router import decide_domain


def dispatch_domain(df):
    """
    v2.x FINAL
    Detect domain and attach execution engine.
    """
    decision = decide_domain(df)
    decision.engine = registry.get_domain(decision.selected_domain)
    return decision
