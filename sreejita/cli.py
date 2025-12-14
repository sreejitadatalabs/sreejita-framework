"""
Sreejita Framework CLI v1.7

Lightweight command-line interface.
Routes execution without triggering heavy modules.
"""

import argparse
import logging
from pathlib import Path
import sys

from sreejita.__version__ import __version__
from sreejita.config.loader import load_config

# automation (safe to import)
from sreejita.automation.batch_runner import run_batch
from sreejita.automation.file_watcher import start_watcher
from sreejita.automation.scheduler import start_scheduler

# reports (single entry)
from sreejita.reports.hybrid import run as run_hybrid

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Sreejita Framework v{__version__} — Data Analytics Automation"
    )

    # positional input (optional)
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV or Excel file"
    )

    # optional config (validated later)
    parser.add_argument(
        "--config",
        required=False,
        help="Path to config YAML"
    )

    # execution modes
    parser.add_argument("--batch", help="Run batch processing on a folder")
    parser.add_argument("--watch", help="Watch a folder for new files")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run batch on a schedule (requires --batch)"
    )

    # meta flags
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # ---- VERSION SHORT-CIRCUIT (CRITICAL FIX) ----
    if args.version:
        print(f"Sreejita Framework v{__version__}")
        return 0

    # ---- LOGGING ----
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # ---- CONFIG VALIDATION (ONLY WHEN NEEDED) ----
    if args.batch or args.watch or args.schedule or args.input:
        if not args.config:
            parser.error("--config is required for execution")

    config = load_config(args.config) if args.config else {}

    # ---- WATCH MODE ----
    if args.watch:
        logger.info("Watching folder: %s", args.watch)
        start_watcher(args.watch, args.config)
        return 0

    # ---- SCHEDULE MODE ----
    if args.schedule:
        if not args.batch:
            parser.error("--schedule requires --batch")
        schedule_cfg = config.get("automation", {}).get("schedule")
        if not schedule_cfg:
            raise ValueError("Missing automation.schedule in config")
        start_scheduler(schedule_cfg, args.batch, args.config)
        return 0

    # ---- BATCH MODE ----
    if args.batch:
        logger.info("Running batch on %s", args.batch)
        run_batch(args.batch, args.config)
        return 0

    # ---- SINGLE FILE MODE ----
    if not args.input:
        parser.error("Input file required unless using --batch or --watch")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    logger.info("Processing file %s", input_path)
    run_hybrid(str(input_path), config)
    logger.info("Completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
