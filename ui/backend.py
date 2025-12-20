from datetime import datetime
from pathlib import Path

import pandas as pd

from sreejita.reports.hybrid import run as run_hybrid
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: str | None = None
) -> dict:
    """
    Streamlit → Framework adapter (v3)

    This function MUST call Hybrid v3.
    NO v1 pipeline allowed here.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision = None
    policy_decision = None

    # -------------------------------------------------
    # Load data (safe)
    # -------------------------------------------------
    try:
        if input_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(input_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, encoding="latin1")
        else:
            df = pd.read_excel(input_path)

        decision = decide_domain(df)

        policy_engine = PolicyEngine(min_confidence=0.7)
        policy_decision = policy_engine.evaluate(decision)

    except Exception as e:
        print("⚠️ Decision/Policy engine failed:", e)

    # -------------------------------------------------
    # 🔥 v3 AUTHORITATIVE PIPELINE
    # -------------------------------------------------
    report_path = run_hybrid(
        input_path=str(input_path),
        config={
            "output_dir": str(output_dir),
            "metadata": {
                "source": "Streamlit UI",
                "domain_hint": domain
            }
        }
    )

    return {
        "report_path": report_path,
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": decision.rules_applied if decision else None,
        "policy_status": policy_decision.status if policy_decision else None,
        "policy_reasons": policy_decision.reasons if policy_decision else None,
        "generated_at": datetime.utcnow().isoformat(),
        "version": "Framework v3.0"
    }
