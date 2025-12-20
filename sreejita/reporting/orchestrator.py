import logging
import pandas as pd
from pathlib import Path

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.domains.router import DOMAIN_IMPLEMENTATIONS

log = logging.getLogger("sreejita.orchestrator")


def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    v3.2 Orchestrator — SINGLE SOURCE OF TRUTH
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ----------------------
    # Load data
    # ----------------------
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    # ----------------------
    # Domain decision
    # ----------------------
    decision = decide_domain(df)
    domain = decision.selected_domain

    engine = DOMAIN_IMPLEMENTATIONS.get(domain)
    if engine is None:
        return {
            "unknown": {
                "kpis": {"rows": len(df), "cols": len(df.columns)},
                "insights": [{
                    "level": "RISK",
                    "title": "Unknown Domain",
                    "so_what": "No matching domain engine found."
                }],
                "recommendations": [],
                "visuals": []
            }
        }

    # ----------------------
    # Domain lifecycle
    # ----------------------
    if hasattr(engine, "preprocess"):
        df = engine.preprocess(df)

    kpis = engine.calculate_kpis(df)
    insights = engine.generate_insights(df, kpis)
    recommendations = enrich_recommendations(
        engine.generate_recommendations(df, kpis)
    )

    # ----------------------
    # Visuals
    # ----------------------
    output_root = Path(config.get("output_dir", "runs"))
    visuals_dir = output_root / input_path.stem / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    visuals = []
    if hasattr(engine, "generate_visuals"):
        visuals = engine.generate_visuals(df, visuals_dir)

    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
        }
    }
