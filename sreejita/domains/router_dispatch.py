from sreejita.domains.registry import registry
from sreejita.domains.router import decide_domain


def dispatch_domain(df):
    decision = decide_domain(df)

    domain_name = decision.selected_domain
    engine = registry.get_domain(domain_name)

    # 🔒 THIS MUST NOW WORK
    decision.engine = engine

    return decision
