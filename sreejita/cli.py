"""Sreejita Framework CLI v1.6

Command-line interface with v1.6 enhancements:
- Data quality validation
- Data profiling
- Dry-run mode
- Metrics collection
- Batch processing
- File watching
- Scheduling
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

from sreejita.config.loader import load_config
from sreejita.reports.hybrid import run as run_hybrid
from sreejita.core.validator import DataQualityValidator
from sreejita.core.profiler import DataProfiler
from sreejita.monitoring.metrics import MetricsCollector
from sreejita.automation.batch_runner import run_batch
from sreejita.automation.file_watcher import start_watcher
from sreejita.automation.scheduler import start_scheduler
from sreejita.database.run_history import RunHistoryDB

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sreejita Framework v1.6 - Universal Data Analytics & Reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sreejita input.csv --config config.yaml                    # Standard run
  sreejita input.csv --config config.yaml --dry-run         # Validation only
  sreejita input.csv --config config.yaml --profile         # Data profile
  sreejita --batch /data/folder --config config.yaml        # Batch processing
  sreejita --watch /data/folder --config config.yaml        # File watching
  sreejita --schedule --batch /data --config config.yaml    # Scheduled batch
        """
    )
    
    # Standard arguments
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV or Excel file path"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Configuration file path (YAML)"
    )
    
    # v1.5 automation arguments
    parser.add_argument(
        "--batch",
        help="Run batch mode on folder containing multiple files"
    )
    parser.add_argument(
        "--watch",
        help="Watch a folder and auto-run on new files"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run batch processing on a schedule (cron-based)"
    )
    
    # v1.6 quality assurance arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validation without writing output (preview mode)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Generate comprehensive data profile report"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data quality validation checks"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on warnings in validation (default: True)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for results"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Collect and display execution metrics"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize run history database
    run_history = RunHistoryDB("runs/history.db")
    
    # Initialize metrics collector if requested
    metrics_collector = None
    if args.metrics:
        metrics_collector = MetricsCollector()
    
    # Handler: File watching
    if args.watch:
        logger.info(f"Starting file watcher on {args.watch}")
        start_watcher(args.watch, args.config)
        return
    
    # Handler: Scheduling
    if args.schedule:
        if not args.batch:
            parser.error("--schedule requires --batch to be set")
        schedule_cfg = config.get("automation", {}).get("schedule")
        if not schedule_cfg:
            raise ValueError("No schedule configuration found in config file")
        logger.info(f"Starting scheduler for batch folder: {args.batch}")
        start_scheduler(
            schedule_config=schedule_cfg,
            input_dir=args.batch,
            config_path=args.config
        )
        return
    
    # Handler: Batch processing
    if args.batch:
        logger.info(f"Running batch processing on folder: {args.batch}")
        run_batch(args.batch, args.config)
        return
    
    # Validate input file is provided
    if not args.input:
        parser.error("input file is required (unless using --batch, --watch, or --schedule)")
    
    # Start metrics collection
    if metrics_collector:
        try:
            df_temp = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_excel(args.input)
            metrics_collector.start(len(df_temp))
        except Exception as e:
            logger.warning(f"Could not start metrics: {e}")
    
    # Handler: Dry-run (validation only)
    if args.dry_run:
        logger.info("Running in DRY-RUN mode (validation only)")
        try:
            df = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_excel(args.input)
            validator = DataQualityValidator(strict_mode=args.strict)
            passed, results = validator.validate(df)
            
            report = validator.get_report()
            print("\n" + "="*60)
            print("DATA QUALITY VALIDATION REPORT")
            print("="*60)
            print(f"Status: {'PASSED ✓' if passed else 'FAILED ✗'}")
            print(f"Timestamp: {report['timestamp']}")
            print(f"Total Checks: {report['total_checks']}")
            print(f"Passed: {report['passed_checks']}")
            print(f"Failed: {report['failed_checks']}")
            print("\nDetailed Results:")
            print(json.dumps(report['results'], indent=2))
            print("="*60)
            
            if not passed:
                exit(1)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            exit(1)
        return
    
    # Handler: Data profiling
    if args.profile:
        logger.info("Generating data profile report")
        try:
            df = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_excel(args.input)
            profiler = DataProfiler()
            profile = profiler.profile(df)
            
            print("\n" + "="*60)
            print("DATA PROFILE REPORT")
            print("="*60)
            print(json.dumps(profile, indent=2, default=str))
            print("="*60)
            
            # Save profile if output specified
            if args.output:
                output_file = Path(args.output) / f"profile_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(profile, f, indent=2, default=str)
                logger.info(f"Profile saved to {output_file}")
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            exit(1)
        return
    
    # Handler: Validation check (without dry-run)
    if args.validate:
        logger.info("Running data quality validation")
        try:
            df = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_excel(args.input)
            validator = DataQualityValidator(strict_mode=args.strict)
            passed, results = validator.validate(df)
            report = validator.get_report()
            print(json.dumps(report, indent=2, default=str))
            if not passed:
                exit(1)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            exit(1)
        return
    
    # Handler: Standard processing (default)
    logger.info(f"Processing file: {args.input}")
    try:
        run_hybrid(args.input, config)
        
        # Record successful run in history
        metrics = metrics_collector.end() if metrics_collector else {}
        run_data = {
            "run_id": f"{Path(args.input).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "input_file": args.input,
            "status": "success",
            "metrics": metrics
        }
        run_history.record_run(run_data)
        
        if args.metrics:
            print("\nExecution Metrics:")
            print(json.dumps(metrics, indent=2))
        
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
        # Record failed run
        if metrics_collector:
            metrics_collector.record_error()
        
        run_data = {
            "run_id": f"{Path(args.input).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "input_file": args.input,
            "status": "failed",
            "errors": 1
        }
        run_history.record_run(run_data)
        exit(1)


if __name__ == "__main__":
    main()
