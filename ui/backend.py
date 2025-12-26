from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from sreejita.cli import run_single_file
from sreejita.config.defaults import DEFAULT_CONFIG


def run_analysis_from_ui(
    input_path: str,
    narrative_enabled: bool = False,
    narrative_provider: str = "gemini",
    generate_pdf: bool = False,
) -> Dict[str, Any]:
    """
    v3.6 UI-safe wrapper around CLI core.

    Returns:
        {
            "html": <path>,
            "pdf": <path or None>,
            "run_dir": <path>
        }
    """

    # -------------------------------------------------
    # Run directory (UI-safe, timestamped)
    # -------------------------------------------------
    run_dir = Path("runs") / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Config (isolated copy)
    # -------------------------------------------------
    config = DEFAULT_CONFIG.copy()
    config["run_dir"] = str(run_dir)
    config["narrative"] = {
        "enabled": narrative_enabled,
        "provider": narrative_provider,
        "confidence_band": "MEDIUM",
    }

    # -------------------------------------------------
    # Delegate to CLI core (v3.6)
    # -------------------------------------------------
    result = run_single_file(
        input_path=input_path,
        config=config,
        generate_pdf=generate_pdf,
    )

    # -------------------------------------------------
    # Stable contract for Streamlit / API
    # -------------------------------------------------
    return {
        "html": result.get("html"),
        "pdf": result.get("pdf"),
        "run_dir": result.get("run_dir"),
    }
