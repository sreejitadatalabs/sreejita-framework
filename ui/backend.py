from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# v1 report pipeline (DO NOT TOUCH)
from sreejita.cli import run_single_file

# v2 decision engine (shadow mode)
from sreejita.domains.router import decide_domain

# v2.5 policy engine (shadow mode)
from sreejita.policy.engine import PolicyEngine

# v3 rendering layer (NEW – SAFE)
from sreejita.reporting.pdf_renderer import PandocPDFRenderer


def run_analysis_from_ui(
    input_path: str,
    domain: str = "Auto",
    output_dir: str = "reports",
    config_path: Optional[str] = None
) -> dict:
    """
    Streamlit → Framework adapter

    Layers:
    - v1.9 : Report generation (authoritative, MUST NOT break)
    - v2.4 : Domain decision intelligence (shadow mode)
    - v2.5 : Policy & governance layer (shadow mode)
    - v3.0 : PDF rendering (optional, delivery layer)

    RULES:
    - v1 pipeline is sacred
    - v2/v2.5 failures must degrade safely
    - PDF rendering must NEVER break analysis
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # -------------------------------------------------
    # v2.4 / v2.5 SHADOW INTELLIGENCE (SAFE)
    # -------------------------------------------------
    decision = None
    policy_decision = None

    try:
        # Robust CSV / Excel loading
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
        print("⚠️ v2/v2.5 engine failed, falling back to v1 only")
        print("Reason:", e)

    # -------------------------------------------------
    # v1.9 REPORT GENERATION (AUTHORITATIVE)
    # -------------------------------------------------
    # This MUST remain untouched
    md_report_path = run_single_file(
        input_path=str(input_path),
        config_path=config_path
    )

    # -------------------------------------------------
    # v3.0 PDF DELIVERY (OPTIONAL & SAFE)
    # -------------------------------------------------
    pdf_report_path = None

    try:
        renderer = PandocPDFRenderer()
        pdf_report_path = renderer.render(
            md_path=Path(md_report_path),
            output_dir=Path(md_report_path).parent
        )
    except Exception as e:
        # PDF must NEVER block analysis
        print("⚠️ PDF generation failed (non-fatal)")
        print("Reason:", e)

    # -------------------------------------------------
    # UNIFIED RETURN PAYLOAD (UI SAFE)
    # -------------------------------------------------
    return {
        # v1 output
        "report_path": md_report_path,
        "pdf_report_path": pdf_report_path,

        # v2.4 decision intelligence
        "domain": decision.selected_domain if decision else domain,
        "domain_confidence": decision.confidence if decision else None,
        "decision_rules": decision.rules_applied if decision else None,
        "decision_fingerprint": getattr(decision, "fingerprint", None),

        # v2.5 policy
        "policy_status": policy_decision.status if policy_decision else None,
        "policy_reasons": policy_decision.reasons if policy_decision else None,

        # metadata
        "generated_at": datetime.utcnow().isoformat(),
        "version": "UI v1.9 / Engine v2.4 / Policy v2.5 / PDF v3.0"
    }
