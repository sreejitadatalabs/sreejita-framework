import os
from datetime import datetime
from pathlib import Path

from sreejita.cli import main as cli_main
from sreejita.config.loader import load_config


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: str | None = None
) -> dict:
    """
    Adapter function for Streamlit UI (v1.9)

    This function:
    - Calls the existing v1.8 pipeline
    - Generates a Hybrid PDF report
    - Returns run metadata for UI display

    IMPORTANT:
    - No analytics logic here
    - No plotting logic here
    """

    # -----------------------------
    # Prepare paths
    # -----------------------------
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_name = f"hybrid_report_{timestamp}.pdf"
    report_path = output_dir / report_name

    # -----------------------------
    # Load config (optional)
    # -----------------------------
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()  # default config

    # Inject domain hint (future v2.x use)
    config["domain"] = domain

    # -----------------------------
    # Call existing CLI pipeline
    # -----------------------------
    # We call CLI programmatically to avoid duplicating logic
    cli_args = [
        "--input", str(input_path),
        "--output", str(report_path)
    ]

    # NOTE:
    # This assumes your CLI main() accepts args injection.
    # If not, we will adapt it cleanly.
    cli_main(cli_args)

    # -----------------------------
    # Collect metadata
    # -----------------------------
    result = {
        "report_path": str(report_path),
        "rows": config.get("run_metadata", {}).get("rows"),
        "columns": config.get("run_metadata", {}).get("columns"),
        "domain": domain,
        "generated_at": datetime.utcnow().isoformat(),
        "version": "1.9.0"
    }

    return result
