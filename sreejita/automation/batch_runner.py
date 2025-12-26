import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from sreejita.reports.hybrid import run as run_hybrid
from sreejita.reporting.html_renderer import HTMLReportRenderer
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger
from sreejita.automation.retry import retry

log = get_logger("batch-runner")

SUPPORTED_EXT = (".csv", ".xlsx")


@retry(times=3, delay=5)
def run_single_file(
    file_path: Path,
    config: dict,
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Process a single file in batch mode.

    v3.3 CONTRACT:
    - Markdown is INTERNAL only
    - HTML is the FINAL output
    - One isolated folder per input file
    - Batch execution must NEVER fail due to rendering
    """

    src = Path(file_path)

    # -------------------------------------------------
    # 1️⃣ Create per-file run folder
    # -------------------------------------------------
    file_run_dir = run_dir / src.stem
    input_dir = file_run_dir / "input"
    failed_dir = file_run_dir / "failed"

    input_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    dst = input_dir / src.name
    dst.write_bytes(src.read_bytes())

    log.info("Processing file: %s", src.name)

    # -------------------------------------------------
    # 2️⃣ Force Hybrid output into THIS folder
    # -------------------------------------------------
    local_config = dict(config)
    local_config["run_dir"] = str(file_run_dir)

    # -------------------------------------------------
    # 3️⃣ Generate Markdown (AUTHORITATIVE, INTERNAL)
    # -------------------------------------------------
    md_path = Path(run_hybrid(str(dst), local_config))

    if not md_path.exists():
        raise RuntimeError(f"Markdown report not created: {md_path}")

    log.info("Markdown generated (internal): %s", md_path)

    # -------------------------------------------------
    # 4️⃣ Render HTML (FINAL OUTPUT)
    # -------------------------------------------------
    html_path = None
    try:
        renderer = HTMLReportRenderer()
        html_path = renderer.render(md_path)
        log.info("HTML report generated: %s", html_path)

    except Exception as e:
        # ABSOLUTE RULE: batch must never fail due to rendering
        log.warning("HTML rendering failed: %s", e)

    log.info("Completed file: %s", src.name)

    return {
        "file": src.name,
        "md_path": str(md_path),
        "html_path": str(html_path) if html_path else None,
        "run_dir": str(file_run_dir),
    }


def run_batch(
    input_folder: str,
    config_path: Optional[str],
    output_root: str = "runs",
):
    """
    Batch processing entry point (v3.3 SAFE).
    """
    config = load_config(config_path)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    log.info("Found %d input files", len(files))

    for file in files:
        src = Path(input_folder) / file

        try:
            run_single_file(src, config, run_dir)

        except Exception as e:
            failed_path = (
                run_dir
                / "failed"
                / f"{src.stem}_{int(datetime.utcnow().timestamp())}{src.suffix}"
            )
            failed_path.write_bytes(src.read_bytes())

            log.error(
                "File failed after retries: %s | Reason: %s",
                src.name,
                str(e),
            )

    log.info("Batch run completed: %s", run_dir)
