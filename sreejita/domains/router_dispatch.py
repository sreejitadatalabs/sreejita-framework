"""
Router dispatcher (v2.x FINAL)

- Detects domain
- Attaches domain engine
- Single source of truth for execution routing
"""

import os
from sreejita.domains.registry import registry
from sreejita.domains.router import decide_domain

# Feature flag (kept for backward compatibility)
USE_ROUTER_V2 = os.getenv("SREEJITA_ROUTER_V2", "false").lower() == "true"


def dispatch_domain(df):
    """
    Detect domain and ATTACH engine.
    This is mandatory for v2.x execution.
    """
    decision = decide_domain(df)

    domain_name = decision.selected_domain
    engine = registry.get_domain(domain_name)

    # 🔒 CRITICAL LINE
    decision.engine = engine

    return decision
