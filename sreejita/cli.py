import argparse
from sreejita.config.loader import load_config
from sreejita.reports.hybrid import run

parser.add_argument("--batch", help="Run batch mode on folder")

def main():
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
