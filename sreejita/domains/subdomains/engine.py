from typing import Dict, List, Any
from sreejita.narrative.executive_cognition import (
    build_executive_payload,
    build_subdomain_executive_payloads,
)

MIN_SUBDOMAIN_CONFIDENCE = 0.30
MIXED_THRESHOLD = 2


class UniversalSubDomainEngine:
    """
    Universal sub-domain intelligence engine.

    Domain-agnostic.
    Deterministic.
    Executive-safe.
    """

    @staticmethod
    def resolve(
        *,
        domain: str,
        signals: Dict[str, float],
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Resolve sub-domains and build executive cognition.

        Returns:
        {
            primary_sub_domain,
            sub_domains,
            executive_by_sub_domain
        }
        """

        # -----------------------------------------
        # 1. Filter Valid Sub-Domains
        # -----------------------------------------
        active = {
            s: round(float(c), 2)
            for s, c in signals.items()
            if isinstance(c, (int, float)) and c >= MIN_SUBDOMAIN_CONFIDENCE
        }

        if not active:
            return {
                "primary_sub_domain": "unknown",
                "sub_domains": {},
                "executive_by_sub_domain": {},
            }

        # -----------------------------------------
        # 2. Primary vs Mixed
        # -----------------------------------------
        primary = max(active, key=active.get)
        is_mixed = len(active) >= MIXED_THRESHOLD

        primary_sub_domain = "mixed" if is_mixed else primary

        # -----------------------------------------
        # 3. Per-Sub-Domain Executive Cognition
        # -----------------------------------------
        executive_by_sub = build_subdomain_executive_payloads(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
        )

        return {
            "primary_sub_domain": primary_sub_domain,
            "sub_domains": active,
            "executive_by_sub_domain": executive_by_sub,
        }
