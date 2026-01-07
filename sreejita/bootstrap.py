"""
Sreejita Bootstrap — Decision Engine Observability
--------------------------------------------------
Authoritative bootstrap for decision-engine observers.

Responsibilities:
- Build observers from config
- Register observers with router safely
- Never crash runtime
- Idempotent (safe to call multiple times)

This file MUST NOT:
- Import domain engines
- Mutate registry
- Assume observer internals
"""

import logging
from typing import Dict, Any, Iterable

from sreejita.domains.router import register_observer
from sreejita.observability.factory import build_observers

log = logging.getLogger("sreejita.bootstrap.decision_engine")


# =====================================================
# SAFE BOOTSTRAP ENTRYPOINT
# =====================================================
def bootstrap_decision_engine(config: Dict[str, Any]) -> None:
    """
    Bootstrap decision-engine observers.

    Args:
        config: Full framework config dictionary

    Guarantees:
    - Never raises
    - Observers registered at most once (router guarded)
    - Missing config handled safely
    """

    try:
        decision_cfg = config.get("decision_engine", {}) if isinstance(config, dict) else {}

        observers: Iterable = build_observers(decision_cfg)

        if not observers:
            log.info("No decision-engine observers configured")
            return

        for observer in observers:
            try:
                register_observer(observer)
                log.info(
                    "✅ Decision-engine observer registered: %s",
                    observer.__class__.__name__,
                )
            except RuntimeError as e:
                # Expected case: duplicate registration
                log.debug(
                    "ℹ️ Observer already registered: %s (%s)",
                    observer.__class__.__name__,
                    str(e),
                )
            except Exception as e:
                log.error(
                    "❌ Failed to register observer: %s | %s",
                    observer.__class__.__name__,
                    str(e),
                    exc_info=True,
                )

    except Exception as e:
        # Absolute safety: bootstrap must NEVER break execution
        log.error(
            "❌ Decision-engine bootstrap failed: %s",
            str(e),
            exc_info=True,
        )
