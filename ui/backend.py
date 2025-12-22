from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# v1 report pipeline (AUTHORITATIVE – DO NOT TOUCH)
from sreejita.cli import run_single_file

# v2 decision engine (shadow mode)
from sreejita.domains.router import decide_domain

# v2.5 policy engine (shadow mode)
from sreejita.policy.engine import PolicyEngine


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",  # fallback label only
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Streamlit → Framework adapter (v3.3 FINAL)

    HARD RULES:
    - v1 pipeline must NEVER break
    - UI must NEVER render Markdown directly
    - HTML is the FINAL client artifact
    - Rendering failures must NEVER block analysis
    """

    input_path = Path(input_path)

    # -------------------------------------------------
    # v2 / v2.5 SHADOW INTELLIGENCE (NON-BLOCKING)
    # -------------------------------------------------
    decision = None
    policy_decision = None

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
        # ABSOLUTE RULE: NEVER BREAK v1
        print("⚠️ Shadow intelligence failed — continuing safely")
        print("Reason:", e)

    # -------------------------------------------------
    # v1 REPORT GENERATION (AUTHORITATIVE)
    # -------------------------------------------------
    # IMPORTANT: run_single_file ALWAYS returns a PATH STRING
    md_report_path = Path(
        run_single_file(
            input_path=str(input_path),
            config_path=config_path,
        )
    )

    report_exists = md_report_path.exists()

    # -------------------------------------------------
    # HTML DISCOVERY (FINAL CLIENT OUTPUT)
    # -------------------------------------------------
    html_report_path: Optional[Path] = None

    if report_exists:
        candidate_html = md_report_path.with_suffix(".html")
        if candidate_html.exists():
            html_report_path = candidate_html
    else:
        print("⚠️ Markdown path returned but file does not exist:", md_report_path)

    # -------------------------------------------------
    # RETURN UI-SAFE PAYLOAD
    # -------------------------------------------------
    return {
        # v1 artifacts
        "md_report_path": str(md_report_path) if md_report_path.exists() else None,
        "html_report_path": str(html_report_path) if html_report_path else None,
        "report_exists": report_exists,

        # v2 decision intelligence (shadow)
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": getattr(decision, "rules_applied", None),
        "decision_fingerprint": getattr(decision, "fingerprint", None),

        # v2.5 policy (shadow)
        "policy_status": getattr(policy_decision, "status", None),
        "policy_reasons": getattr(policy_decision, "reasons", None),

        # metadata
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v3.3 / Engine v3.3 / HTML Renderer v1.0",
    }
