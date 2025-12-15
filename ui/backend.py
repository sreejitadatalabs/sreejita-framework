from datetime import datetime
from pathlib import Path

from sreejita.cli import run_single_file


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: str | None = None
) -> dict:
    """
    Streamlit → v1.8 backend adapter (v1.9)

    Calls the programmatic pipeline directly.
    No CLI usage. No arg parsing.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run analysis
    report_path = run_single_file(
        input_path=str(input_path),
        config_path=config_path
    )

    return {
        "report_path": report_path,
        "domain": domain,
        "generated_at": datetime.utcnow().isoformat(),
        "version": "1.9.0"
    }
