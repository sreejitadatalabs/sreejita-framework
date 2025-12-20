from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional  # Changed: Added Optional

class BaseReport(ABC):
    """
    Abstract Base Class for all Report Engines.
    Ensures every report type (Hybrid, PDF, CSV) follows the same contract.
    """

    name = "base"

    @abstractmethod
    def build(
        self,
        domain_results: Dict[str, Any],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None  # Changed: | None -> Optional[...]
    ) -> Path:
        """
        Generates the report artifact.
        
        Args:
            domain_results: Dictionary containing data from all analyzed domains.
            output_dir: Path object where the report should be saved.
            metadata: Optional dictionary for extra context (e.g., user info).
            
        Returns:
            Path object to the generated report file.
        """
        pass
