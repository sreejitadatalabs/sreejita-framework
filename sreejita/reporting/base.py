from pathlib import Path
from typing import Dict, Any, List


class BaseReport:
    """
    BaseReport v3.x
    ----------------
    Contract for all report engines (Hybrid, Executive, Dynamic).

    Responsibilities:
    - Define build() interface
    - Enforce consistent output handling
    - Provide shared helpers if needed later
    """

    name: str = "base"

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Dict[str, Any] | None = None
    ) -> Path:
        """
        Build the report.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build()"
        )

    # -----------------------------
    # Optional shared helpers
    # -----------------------------

    def validate_results(self, domain_results: Dict[str, Dict[str, Any]]):
        """
        Basic sanity validation for domain outputs.
        """
        if not isinstance(domain_results, dict):
            raise ValueError("domain_results must be a dict")

        for domain, result in domain_results.items():
            if not isinstance(result, dict):
                raise ValueError(f"Result for domain '{domain}' must be a dict")
