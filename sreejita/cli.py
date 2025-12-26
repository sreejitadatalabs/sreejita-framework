"""
Sreejita Framework CLI
v3.6 — HTML Primary + Optional Chromium PDF
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime
import importlib

from sreejita.__version__ import __version__
from sreejita.config.loader import load_config

# automation (safe to import)
from sreejita.automation.batch_runner import run_batch
from sreejita.automation.file_watcher import start_watcher
from sreejita.automation.scheduler import start_scheduler

logger = logging.getLogger(__name__)


# -------------------------------------------------
# PROGRAMMATIC ENTRY (CLI / Streamlit / API)
# -------------------------------------------------
def run_single_file(
    input_path: str,
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    generate_pdf: bool = False,
) -> Dict[str, Optional[str]]:
    """
    v3.6 Programmatic Entry

    Returns:
        {
            "html": <path>,
            "pdf": <path or None>,
            "run_dir": <path>
        }
    """

    # ---- Lazy imports (critical) ----
    importlib.import_module("sreejita.domains.bootstrap_v2")

    hybrid = importlib.import_module("sreejita.reporting.hybrid")
    html_renderer_mod = importlib.import_module(
        "sreejita.reporting.html_renderer"
    )

    # ---- Run directory ----
    run_dir = Path("runs") / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config ----
    final_config = config or load_config(config_path)
    final_config["run_dir"] = str(run_dir)

    # ---- Generate Markdown ----
    md_path = Path(hybrid.run(input_path, final_config))

    # ---- Generate HTML (PRIMARY OUTPUT) ----
    html_renderer = html_renderer_mod.HTMLReportRenderer()
    html_path = html_renderer.render(md_path, output_dir=run_dir)

    pdf_path = None

    # ---- Optional PDF (v3.6) ----
    if generate_pdf:
        try:
            pdf_renderer_mod = importlib.import_module(
                "sreejita.reporting.pdf_renderer"
            )
            pdf_renderer = pdf_renderer_mod.PDFRenderer()
            pdf_path = pdf_renderer.render(html_path, output_dir=run_dir)
        except Exception as e:
            logger.warning("PDF generation failed (non-blocking): %s", e)

    return {
        "html": str(html_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "run_dir": str(run_dir),
    }


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
    parser.add_argument("--schedule", action="store_true")

    parser.add_argument("--pdf", action="store_true", help="Export PDF (v3.6)")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    # ---- VERSION ----
    if args.version:
        print(f"Sreejita Framework v{__version__}")
        return 0

    # ---- LOGGING ----
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
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
            args.config,
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

    result = run_single_file(
        input_path=str(input_path),
        config_path=args.config,
        generate_pdf=args.pdf,
    )

    print("\n✅ Report generated successfully")
    print(f"🌐 HTML Report: {result['html']}")

    if result["pdf"]:
        print(f"📄 PDF Report:  {result['pdf']}")

    print(f"📁 Run folder:  {result['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
