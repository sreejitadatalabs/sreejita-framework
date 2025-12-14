import os
from pathlib import Path
from datetime import datetime

from sreejita.reports.hybrid import run as run_hybrid
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger
from sreejita.automation.retry import retry

log = get_logger("batch-runner")

SUPPORTED_EXT = (".csv", ".xlsx")

@retry(times=3, delay=5)


@retry(times=3, delay=5)
def run_single_file(file_path, config_path, output_root="runs"):
    config = load_config(config_path)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / timestamp
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    failed_dir = run_dir / "failed"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)
    dst = input_dir / src.name
    dst.write_bytes(src.read_bytes())

    try:
        log.info("Processing file: %s", src.name)
        run_hybrid(str(dst), config)
        log.info("Completed file: %s", src.name)

    except Exception as e:
        failed_dir.mkdir(exist_ok=True)
        failed_path = failed_dir / src.name
        failed_path.write_bytes(src.read_bytes())

        log.error(
            "File failed and moved to failed/: %s | Reason: %s",
            src.name,
            str(e)
        )

        # Important: do NOT crash entire batch
        return
 


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
                    try:
                        src = Path(input_folder) / file
            run_single_file(src, config_path)
            except Exception:
            log.warning("Skipping failed file and continuing batch")

    log.info("Batch run completed: %s", run_dir)

