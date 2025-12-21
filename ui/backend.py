from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# v1 report pipeline (AUTHORITATIVE – DO NOT TOUCH)
from sreejita.cli import run_single_file

# v2 decision engine (shadow mode)
from sreejita.domains.router import decide_domain

# v2.5 policy engine (shadow mode)
from sreejita.policy.engine import PolicyEngine


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: Optional[str] = None,
) -> dict:
    """
    Streamlit → Framework adapter (v3.3 SAFE)

    Layers:
    - v1.x : Report generation (AUTHORITATIVE)
    - v2.x : Domain intelligence (SHADOW)
    - v2.5 : Policy engine (SHADOW)
    - v3.x : PDF delivery (PASSIVE ONLY)

    HARD RULES:
    - v1 pipeline must NEVER break
    - UI must NEVER require Pandoc
    - PDFs are CONSUMED, not GENERATED, here
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # IMPORTANT: run_single_file returns a PATH STRING
    report_path = Path(
        run_single_file(
            input_path=str(input_path),
            config_path=config_path,
        )
    )

    # -------------------------------------------------
    # SAFE PATH RESOLUTION (MD / PDF)
    # -------------------------------------------------
    md_report_path = None
    pdf_report_path = None

    if report_path.exists():
        if report_path.suffix.lower() == ".pdf":
            pdf_report_path = report_path
            candidate_md = report_path.with_suffix(".md")
            if candidate_md.exists():
                md_report_path = candidate_md
        elif report_path.suffix.lower() == ".md":
            md_report_path = report_path
            candidate_pdf = report_path.with_suffix(".pdf")
            if candidate_pdf.exists():
                pdf_report_path = candidate_pdf
    else:
        print("⚠️ Report path returned but file does not exist:", report_path)

    # -------------------------------------------------
    # RETURN UI-SAFE PAYLOAD
    # -------------------------------------------------
    return {
        # v1 output
        "md_report_path": str(md_report_path) if md_report_path else None,
        "pdf_report_path": str(pdf_report_path) if pdf_report_path else None,

        # v2 decision intelligence
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": getattr(decision, "rules_applied", None),
        "decision_fingerprint": getattr(decision, "fingerprint", None),

        # v2.5 policy
        "policy_status": getattr(policy_decision, "status", None),
        "policy_reasons": getattr(policy_decision, "reasons", None),

        # metadata
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v1.x / Engine v2.x / Policy v2.5 / PDF v3.x",
    }
