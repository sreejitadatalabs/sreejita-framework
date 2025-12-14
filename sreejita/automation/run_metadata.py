import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def create_run_metadata(
    input_files: List[str],
    config: Dict,
    output_dir: Path,
    status: str = "completed",
    errors: List[str] | None = None,
):
    """
    Create a run.json metadata file describing a batch or single-file run.
    """

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "input_files": input_files,
        "errors": errors or [],
        "config_summary": list(config.keys()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path
