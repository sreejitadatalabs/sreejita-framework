import argparse
from sreejita.config.loader import load_config
from sreejita.reports.dynamic import run_dynamic
from sreejita.reports.hybrid import run_hybrid
from sreejita.reports.executive import run_executive

def main():
    parser = argparse.ArgumentParser("Sreejita Framework v1.1")
    parser.add_argument("input")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dynamic","hybrid","executive"])
    parser.add_argument("--output", default="report.pdf")

    args = parser.parse_args()
    cfg = load_config(args.config)

    mode = args.mode or cfg["report"]["mode"]

    if mode == "dynamic":
        run_dynamic(args.input, args.output, cfg)
    elif mode == "hybrid":
        run_hybrid(args.input, args.output, cfg)
    else:
        run_executive(args.input, args.output, cfg)
