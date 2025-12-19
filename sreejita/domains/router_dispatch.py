"""
Router dispatcher with feature flag.

Controls whether v1 or v2 routing is used.
Default: v1 (safe)
"""

import os

# Feature flag
USE_ROUTER_V2 = os.getenv("SREEJITA_ROUTER_V2", "false").lower() == "true"


def apply_domain(df, domain_name=None):
    """
    Unified apply_domain entry point.

    - v1 requires domain_name
    - v2 auto-detects domain
    """

    if USE_ROUTER_V2:
        # v2 path (auto-detection)
        from sreejita.domains.router_v2 import apply_domain as apply_v2
        return apply_v2(df)

    else:
        # v1 path (explicit domain)
        from sreejita.domains.router import apply_domain as apply_v1

        if domain_name is None:
            raise ValueError(
                "domain_name is required when using v1 router "
                "(set SREEJITA_ROUTER_V2=true to enable v2)"
            )

        return apply_v1(df, domain_name)
