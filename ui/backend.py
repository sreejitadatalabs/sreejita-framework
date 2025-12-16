from datetime import datetime
from pathlib import Path

import pandas as pd

# v1 report pipeline (DO NOT TOUCH)
from sreejita.cli import run_single_file

# v2 decision engine
from sreejita.domains.router import decide_domain

# v2.5 policy engine
from sreejita.policy.engine import PolicyEngine


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: str | None = None
) -> dict:
    """
    Streamlit → Framework adapter

    Layers:
    - v1.9 : Report generation (stable, authoritative)
    - v2.4 : Domain decision intelligence (shadow mode)
    - v2.5 : Policy & governance layer (shadow mode)

    IMPORTANT:
    - v1 report MUST NEVER break
    - v2/v2.5 failures must degrade safely
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # -------------------------------------------------
    # v2.4 Domain Intelligence (SAFE / SHADOW MODE)
    # -------------------------------------------------
    decision = None
    policy_decision = None

    try:
        # Robust CSV / Excel loading (encoding-safe)
        if input_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(input_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding="latin1")
        else:
            df = pd.read_excel(input_path)

        # Run domain decision engine
        decision = decide_domain(df)

        # -------------------------------------------------
        # v2.5 Policy Evaluation
        # -------------------------------------------------
        policy_engine = PolicyEngine(min_confidence=0.7)
        policy_decision = policy_engine.evaluate(decision)

    except Exception as e:
        # ABSOLUTE RULE:
        # v2 / v2.5 must NEVER break v1 demo
        print("⚠️ v2/v2.5 engine failed, falling back to v1 only")
        print("Reason:", e)

    # -------------------------------------------------
    # v1.9 Report Generation (AUTHORITATIVE)
    # -------------------------------------------------
    report_path = run_single_file(
        input_path=str(input_path),
        config_path=config_path
    )

    # -------------------------------------------------
    # Unified Return Payload (UI-safe)
    # -------------------------------------------------
    return {
        # v1 output
        "report_path": report_path,

        # v2.4 decision (if available)
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": decision.rules_applied if decision else None,
        "decision_fingerprint": getattr(decision, "fingerprint", None),

        # v2.5 policy result (if available)
        "policy_status": policy_decision.status if policy_decision else None,
        "policy_reasons": policy_decision.reasons if policy_decision else None,

        # run metadata
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v1.9 / Engine v2.4 / Policy v2.5"
    }
