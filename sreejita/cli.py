import argparse
from sreejita.config.loader import load_config
from sreejita.reports.hybrid import run

if args.schedule:
    from sreejita.automation.scheduler import start_scheduler

    config = load_config(args.config)
    schedule_cfg = config.get("automation", {}).get("schedule")

    if not schedule_cfg:
        raise ValueError("No schedule configuration found in config file")

    start_scheduler(
        schedule_config=schedule_cfg,
        input_dir=args.batch,
        config_path=args.config
    )
    return

parser.add_argument("--batch", help="Run batch mode on folder")
parser.add_argument(
    "--watch",
    help="Watch a folder and auto-run on new files"
)


def main():
    if args.watch:
    from sreejita.automation.file_watcher import start_watcher
    start_watcher(args.watch, args.config)
    return

    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run batch processing on a schedule (cron-based)"
    )

    parser = argparse.ArgumentParser("Sreejita Framework v1.2")
    parser.add_argument("input")
    parser.add_argument("--config", required=True)
    

    args = parser.parse_args()
    config = load_config(args.config)
    run(args.input, config)
    
    if args.batch:
    from sreejita.automation.batch_runner import run_batch
    run_batch(args.batch, args.config)
    return
