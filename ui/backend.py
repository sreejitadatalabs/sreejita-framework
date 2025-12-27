from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from sreejita.cli import run_single_file
from sreejita.config.defaults import DEFAULT_CONFIG


def run_analysis_from_ui(
    input_path: str,
    narrative_enabled: bool = False,
    narrative_provider: str = "gemini",
    generate_pdf: bool = True,  # PDF is DEFAULT in v3.5.1
) -> Dict[str, Any]:
    """
    v3.5.1 UI-safe wrapper (STABLE)

    ReportLab-only pipeline.
    No HTML.
    No browsers.
    Guaranteed PDF.

    Returns:
        {
            "pdf": <path>,
            "markdown": <path>,
            "run_dir": <path>
        }
    """

    # -------------------------------------------------
    # Run directory (Streamlit-safe)
    # -------------------------------------------------
    run_dir = Path("runs") / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Config (isolated, authoritative)
    # -------------------------------------------------
    config = DEFAULT_CONFIG.copy()
    config["run_dir"] = str(run_dir)
    config["narrative"] = {
        "enabled": narrative_enabled,
        "provider": narrative_provider,
        "confidence_band": "MEDIUM",
    }

    # -------------------------------------------------
    # Delegate to CLI (PDF PRIMARY)
    # -------------------------------------------------
    result = run_single_file(
        input_path=input_path,
        config=config,
        generate_pdf=True,
    )

    # -------------------------------------------------
    # Stable contract for Streamlit
    # -------------------------------------------------
    return {
    "markdown": result.get("markdown"),
    "pdf": result.get("pdf"),
    "run_dir": result.get("run_dir"),
    }

