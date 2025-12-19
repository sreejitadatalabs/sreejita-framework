"""
Sreejita Framework CLI v1.9
Unified CLI + Programmatic Entry Point
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

from sreejita.__version__ import __version__
from sreejita.config.loader import load_config

# automation
from sreejita.automation.batch_runner import run_batch
from sreejita.automation.file_watcher import start_watcher
from sreejita.automation.scheduler import start_scheduler

# reports
# 🔥 DOMAIN BOOTSTRAP — MUST BE FIRST
from sreejita.domains.bootstrap_v2 import *  # noqa: F401

from sreejita.reports.hybrid import run as run_hybrid

logger = logging.getLogger(__name__)


# -------------------------------------------------
# PROGRAMMATIC ENTRY (USED BY STREAMLIT, API, ETC.)
# -------------------------------------------------
def run_single_file(
    input_path: str,
    config_path: Optional[str] = None
) -> str:
    """
    Programmatic execution for UI / API use.
    Returns generated report path.
    """
    config = load_config(config_path) if config_path else {}
    report_path = run_hybrid(input_path, config)
    return report_path


# -------------------------------------------------
# CLI ENTRY
# -------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Sreejita Framework v{__version__} — Data Analytics Automation"
    )

    parser.add_argument("input", nargs="?", help="Input CSV or Excel file")
    parser.add_argument("--config", required=False, help="Path to config YAML")

    parser.add_argument("--batch", help="Run batch processing on a folder")
    parser.add_argument("--watch", help="Watch a folder for new files")
    parser.add_argument("--schedule", action="store_true", help="Run batch on a schedule")

    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    # ---- VERSION ----
    if args.version:
        print(f"Sreejita Framework v{__version__}")
        return 0

    # ---- LOGGING ----
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # ---- CONFIG VALIDATION ----
    if (args.batch or args.watch or args.schedule or args.input) and not args.config:
        parser.error("--config is required for execution")

    config = load_config(args.config) if args.config else {}

    # ---- WATCH ----
    if args.watch:
        start_watcher(args.watch, args.config)
        return 0

    # ---- SCHEDULE ----
    if args.schedule:
        if not args.batch:
            parser.error("--schedule requires --batch")
        start_scheduler(
            config.get("automation", {}).get("schedule"),
            args.batch,
            args.config
        )
        return 0

    # ---- BATCH ----
    if args.batch:
        run_batch(args.batch, args.config)
        return 0

    # ---- SINGLE FILE ----
    if not args.input:
        parser.error("Input file required unless using --batch or --watch")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    logger.info("Processing file %s", input_path)
    report_path = run_hybrid(str(input_path), config)

    print("\n✅ Report generated successfully")
    print(f"📄 Report location: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
