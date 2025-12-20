import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from sreejita.reports.hybrid import run as run_hybrid
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger
from sreejita.automation.retry import retry
from sreejita.domains.router import decide_domain

# 🔹 NEW: PDF Renderer
from sreejita.reporting.pdf_renderer import PandocPDFRenderer

log = get_logger("batch-runner")

SUPPORTED_EXT = (".csv", ".xlsx")


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


@retry(times=3, delay=5)
def run_single_file(file_path: Path, config: dict, run_dir: Path):
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    failed_dir = run_dir / "failed"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)
    dst = input_dir / src.name
    dst.write_bytes(src.read_bytes())

    log.info("Processing file: %s", src.name)

    # 1️⃣ Load data
    df = _load_dataframe(dst)

    # 2️⃣ Domain decision
    decision = decide_domain(df)

    domain_results = {
        decision.selected_domain: {
            "kpis": decision.kpis,
            "insights": decision.insights,
            "recommendations": decision.recommendations,
            "visuals": decision.visuals,
        }
    }

    # 3️⃣ Generate Markdown report
    md_path = run_hybrid(
        domain_results=domain_results,
        output_dir=output_dir,
        metadata={
            "source_file": src.name,
            "domain": decision.selected_domain,
            "confidence": f"{decision.confidence:.2f}",
        },
    )

    log.info("Markdown report generated: %s", md_path)

    # 4️⃣ OPTIONAL: Generate PDF
    if config.get("export_pdf", True):
        try:
            renderer = PandocPDFRenderer()
            pdf_path = renderer.render(md_path)
            log.info("PDF report generated: %s", pdf_path)
        except Exception as e:
            log.warning("PDF generation failed: %s", e)

    log.info("Completed file: %s", src.name)


def run_batch(input_folder: str, config_path: Optional[str], output_root="runs"):
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
            failed_path = run_dir / "failed" / src.name
            failed_path.write_bytes(src.read_bytes())
            log.error(
                "File failed after retries: %s | Reason: %s",
                src.name,
                str(e),
            )

    log.info("Batch run completed: %s", run_dir)
