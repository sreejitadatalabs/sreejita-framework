import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from sreejita.reports.hybrid import run as run_hybrid
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
):
    """
    Process a single file in batch mode.

    v3.3 CONTRACT:
    - Generates Markdown report (authoritative)
    - PDF generation is OPTIONAL and SAFE
    - Never breaks batch execution
    """

    input_dir = run_dir / "input"
    failed_dir = run_dir / "failed"

    input_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)
    dst = input_dir / src.name
    dst.write_bytes(src.read_bytes())

    log.info("Processing file: %s", src.name)

    # -------------------------------------------------
    # 1️⃣ FORCE reports into THIS batch run directory
    # -------------------------------------------------
    local_config = dict(config)
    local_config["output_dir"] = str(run_dir)

    # -------------------------------------------------
    # 2️⃣ Generate Markdown via Hybrid (v3.3)
    # -------------------------------------------------
    md_path = Path(run_hybrid(str(dst), local_config))

    log.info("Markdown report generated: %s", md_path)

    # -------------------------------------------------
    # 3️⃣ OPTIONAL: Generate PDF (safe, isolated)
    # -------------------------------------------------
    if local_config.get("export_pdf", False):
        try:
            from sreejita.reporting.pdf_renderer import PandocPDFRenderer

            renderer = PandocPDFRenderer()
            pdf_path = renderer.render(md_path)
            log.info("PDF report generated: %s", pdf_path)

        except Exception as e:
            log.warning("PDF generation skipped: %s", e)

    log.info("Completed file: %s", src.name)


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
            # Preserve failed file with timestamp suffix
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
