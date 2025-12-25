from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from sreejita.cli import run_single_file
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    narrative_enabled: bool = False,
    narrative_provider: str = "gemini",
) -> Dict[str, Any]:

    input_path = Path(input_path)

    # -----------------------------
    # Shadow intelligence (safe)
    # -----------------------------
    decision = None
    policy_decision = None

    try:
        df = (
            pd.read_csv(input_path, encoding="utf-8")
            if input_path.suffix.lower() == ".csv"
            else pd.read_excel(input_path)
        )
        decision = decide_domain(df)
        policy_decision = PolicyEngine(0.7).evaluate(decision)
    except Exception as e:
        print("Shadow intelligence failed:", e)

    # -----------------------------
    # Runtime config (v3.5)
    # -----------------------------
    runtime_config = {
        "narrative": {
            "enabled": narrative_enabled,
            "provider": narrative_provider,
            "model": "gemini-1.5-flash" if narrative_provider == "gemini" else "gpt-4o-mini",
            "confidence_band": "MEDIUM",
        }
    }

    md_path = Path(
        run_single_file(
            input_path=str(input_path),
            runtime_config=runtime_config,
        )
    )

    html_path = md_path.with_suffix(".html") if md_path.exists() else None

    return {
        "md_report_path": str(md_path) if md_path.exists() else None,
        "html_report_path": str(html_path) if html_path and html_path.exists() else None,
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": getattr(decision, "rules_applied", None),
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v3.5 / Engine v3.5",
    }
