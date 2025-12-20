from typing import Dict, Any, Optional
from pathlib import Path


class BaseReport:
    """
    Base class for all report engines.
    Provides a common interface enforced by the framework.
    """

    name: str = "base"

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Build a report and return the output path.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Report engines must implement build()")
