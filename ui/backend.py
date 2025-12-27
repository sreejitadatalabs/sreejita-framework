from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import copy

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

    - Markdown source of truth
    - ReportLab PDF (guaranteed)
    - No HTML
    - No browser dependencies
    """

    # -------------------------------------------------
    # Run directory (Streamlit-safe)
    # -------------------------------------------------
    run_dir = Path("runs") / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # CONFIG (DEEP COPY — CRITICAL FIX)
    # -------------------------------------------------
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["run_dir"] = str(run_dir)
    config["narrative"]["enabled"] = narrative_enabled
    config["narrative"]["provider"] = narrative_provider

    # -------------------------------------------------
    # Delegate to CLI (PDF PRIMARY)
    # -------------------------------------------------
    result = run_single_file(
        input_path=input_path,
        config=config,
        generate_pdf=generate_pdf,
    )

    # -------------------------------------------------
    # HARD VALIDATION (NO SILENT FAILURES)
    # -------------------------------------------------
    markdown = result.get("markdown")
    pdf = result.get("pdf")

    if not markdown or not Path(markdown).exists():
        raise RuntimeError("Markdown report was not generated")

    if generate_pdf and (not pdf or not Path(pdf).exists()):
        raise RuntimeError(
            "PDF generation failed — check pdf_renderer or payload contract"
        )

    # -------------------------------------------------
    # Stable contract for Streamlit
    # -------------------------------------------------
    return {
        "markdown": markdown,
        "pdf": pdf,
        "run_dir": str(run_dir),
    }
