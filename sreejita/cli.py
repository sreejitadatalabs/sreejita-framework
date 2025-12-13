import argparse
from sreejita.config.loader import load_config
from sreejita.reports.hybrid import run

def main():
    parser = argparse.ArgumentParser("Sreejita Framework v1.2")
    parser.add_argument("input")
    parser.add_argument("--config", required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    run(args.input, config)
