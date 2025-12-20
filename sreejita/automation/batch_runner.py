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

log = get_logger("batch-runner")

SUPPORTED_EXT = (".csv", ".xlsx")


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() == ".xlsx":
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

    # 2️⃣ Decide domain + run domain engine
    domain_result = decide_domain(df)

    domain_results = {
        domain_result.domain: {
            "kpis": domain_result.kpis,
            "insights": domain_result.insights,
            "recommendations": domain_result.recommendations,
            "visuals": domain_result.visuals,
        }
    }

    # 3️⃣ Run Hybrid Report (CORRECT)
    run_hybrid(
        domain_results=domain_results,
        output_dir=output_dir,
        metadata={
            "source_file": src.name,
            "domain": domain_result.domain,
            "confidence": f"{domain_result.confidence:.2f}",
        },
    )

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
