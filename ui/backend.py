from datetime import datetime
from pathlib import Path
from sreejita.cli import run_single_file
from sreejita.config.defaults import DEFAULT_CONFIG


def run_analysis_from_ui(input_path: str, narrative_enabled=False, narrative_provider="gemini"):
    run_dir = Path("runs") / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG.copy()
    config["run_dir"] = str(run_dir)
    config["narrative"] = {
        "enabled": narrative_enabled,
        "provider": narrative_provider,
        "confidence_band": "MEDIUM",
    }

    html_path = run_single_file(input_path, config=config)

    return {
        "html_report_path": html_path,
        "run_dir": str(run_dir),
    }
