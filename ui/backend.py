from datetime import datetime
from pathlib import Path

import pandas as pd

from sreejita.cli import run_single_file
from sreejita.domains.router import decide_domain


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: str | None = None
) -> dict:
    """
    Streamlit → v1 backend adapter (v1.9)
    + v2.4 Domain Intelligence (shadow mode)

    - v1 report remains unchanged
    - v2 decision runs in parallel
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # -----------------------------
    # v2.4 Domain Intelligence
    # -----------------------------
    try:
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        else:
            df = pd.read_excel(input_path)

        decision = decide_domain(df)

    except Exception as e:
        raise RuntimeError("v2 decision engine failed") from e


    # -----------------------------
    # v1.9 Report Generation
    # -----------------------------
    report_path = run_single_file(
        input_path=str(input_path),
        config_path=config_path
    )

    # -----------------------------
    # Return combined metadata
    # -----------------------------
    return {
        "report_path": report_path,

        # v2 decision (shadow)
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": decision.rules_applied if decision else None,
        "decision_fingerprint": getattr(decision, "fingerprint", None),

        # run metadata
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v1.9 / Engine v2.4"
    }
