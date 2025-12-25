from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from sreejita.cli import run_single_file
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.config.defaults import DEFAULT_CONFIG


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    narrative_enabled: bool = False,
    narrative_provider: str = "gemini",
) -> Dict[str, Any]:

    input_path = Path(input_path)

    # -------------------------------------------------
    # Shadow intelligence (non-blocking)
    # -------------------------------------------------
    decision = None
    try:
        df = (
            pd.read_csv(input_path, encoding="utf-8")
            if input_path.suffix.lower() == ".csv"
            else pd.read_excel(input_path)
        )
        decision = decide_domain(df)
        PolicyEngine(0.7).evaluate(decision)
    except Exception as e:
        print("Shadow intelligence failed:", e)

    # -------------------------------------------------
    # Build config SAFELY (v3.5)
    # -------------------------------------------------
    config = DEFAULT_CONFIG.copy()

    config["narrative"] = {
        "enabled": narrative_enabled,
        "provider": narrative_provider,
        "model": (
            "gemini-1.5-flash"
            if narrative_provider == "gemini"
            else "gpt-4o-mini"
        ),
        "confidence_band": "MEDIUM",
    }

    # -------------------------------------------------
    # Call framework (NO new arguments)
    # -------------------------------------------------
    md_report_path = Path(
        run_single_file(
            input_path=str(input_path),
            config=config,
        )
    )

    html_report_path = (
        md_report_path.with_suffix(".html")
        if md_report_path.exists()
        else None
    )

    return {
        "md_report_path": str(md_report_path),
        "html_report_path": str(html_report_path)
        if html_report_path and html_report_path.exists()
        else None,
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v3.5 / Engine v3.5",
    }
