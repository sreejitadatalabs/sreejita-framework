import os
from pathlib import Path
from datetime import datetime

from sreejita.reports.hybrid import run as run_hybrid
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger

log = get_logger("batch-runner")

SUPPORTED_EXT = (".csv", ".xlsx")

def run_batch(input_folder: str, config_path: str, output_root="runs"):
    config = load_config(config_path)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / timestamp
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    log.info("Found %d input files", len(files))

    for file in files:
        src = Path(input_folder) / file
        dst = input_dir / file
        dst.write_bytes(src.read_bytes())

        out_pdf = output_dir / f"{file.split('.')[0]}_report.pdf"
        log.info("Processing %s", file)

        run_hybrid(str(dst), config)

    log.info("Batch run completed: %s", run_dir)

