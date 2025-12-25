import logging
from pathlib import Path
import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations

log = logging.getLogger("sreejita.orchestrator")


def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    v3.3 Orchestrator — SINGLE SOURCE OF TRUTH

    Responsibilities:
    - Load data
    - Decide domain
    - Execute domain lifecycle
    - Return a STANDARDIZED payload
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # -------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------
    if input_path.suffix.lower() == ".csv":
        def _read_csv_safe(path: str):
            try:
                return pd.read_csv(path)
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(path, encoding="latin-1")
                except UnicodeDecodeError:
                    return pd.read_csv(path, encoding="cp1252")

    elif input_path.suffix.lower() in (".xls", ".xlsx"):
        try:
            return pd.read_excel(path)
        except ValueError:
            # Fallback for old or weird Excel files
            try:
                return pd.read_excel(path, engine="openpyxl")
            except Exception:
                return pd.read_excel(path, engine="xlrd")
    
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    # -------------------------------------------------
    # 2. DOMAIN DECISION (AUTHORITATIVE)
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    if engine is None:
        log.warning("No engine found for domain: %s", domain)
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

    # -------------------------------------------------
    # 3. DOMAIN LIFECYCLE
    # -------------------------------------------------
    if hasattr(engine, "preprocess"):
        df = engine.preprocess(df)

    kpis = engine.calculate_kpis(df)
    insights = engine.generate_insights(df, kpis)

    raw_recommendations = engine.generate_recommendations(df, kpis)
    recommendations = enrich_recommendations(raw_recommendations)

    # -------------------------------------------------
    # 4. VISUALS (REPORT-RELATIVE STRATEGY)
    # -------------------------------------------------
    # IMPORTANT:
    # The orchestrator does NOT decide run_dir.
    # HybridReport controls final output location.
    #
    # Strategy:
    # - Place visuals in a TEMP folder under output_dir/visuals
    # - HybridReport will link them by filename
    #
    output_root = Path(config.get("output_dir", "runs"))
    visuals_dir = output_root / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    visuals = []
    if hasattr(engine, "generate_visuals"):
        try:
            visuals = engine.generate_visuals(df, visuals_dir)
        except Exception as e:
            log.warning("Visual generation failed: %s", e)

    # -------------------------------------------------
    # 5. STANDARDIZED PAYLOAD
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
        }
    }
