import logging
import pandas as pd
from pathlib import Path

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations

# Initialize Logger
log = logging.getLogger("sreejita.orchestrator")

def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    v3.2 ORCHESTRATOR — INTEGRATED LIFECYCLE
    
    Responsibilities:
    - Load & validate data
    - Route to correct domain engine
    - Execute analysis lifecycle (KPIs -> Insights -> Visuals)
    - Scopes visuals to the specific input file to prevent collisions
    """

    # -------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
    except Exception as e:
        log.error(f"Data ingestion failed: {e}")
        raise ValueError(f"Data ingestion failed: {str(e)}")

    # -------------------------------------------------
    # 2. DOMAIN ROUTING
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    # Fallback for unknown data
    if engine is None:
        log.warning(f"No matching domain found for {input_path.name}")
        return {
            "unknown": {
                "kpis": {"rows": len(df), "cols": len(df.columns)},
                "insights": [{
                    "level": "RISK",
                    "title": "Unrecognized Data Structure",
                    "so_what": "The system could not map this dataset to a known business domain."
                }],
                "recommendations": [{
                    "action": "Check data schema against supported domain templates",
                    "priority": "HIGH",
                    "timeline": "Immediate"
                }],
                "visuals": []
            }
        }

    # -------------------------------------------------
    # 3. DOMAIN LIFECYCLE EXECUTION
    # -------------------------------------------------
    
    # A. Preprocessing
    if hasattr(engine, "preprocess"):
        df = engine.preprocess(df)

    # B. Analysis (KPIs & Insights)
    kpis = engine.calculate_kpis(df) if hasattr(engine, "calculate_kpis") else {}
    insights = engine.generate_insights(df, kpis) if hasattr(engine, "generate_insights") else []
    
    # C. Recommendations (Enriched)
    raw_recs = engine.generate_recommendations(df, kpis) if hasattr(engine, "generate_recommendations") else []
    recommendations = enrich_recommendations(raw_recs)

    # D. Visuals (Run-Scoped)
    # Strategy: Use input filename stem to isolate visuals for this specific run.
    output_root = Path(config.get("output_dir", "reports"))
    run_id = input_path.stem  # e.g., "sales_data_q3" from "sales_data_q3.csv"
    
    visuals_dir = output_root / run_id / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    visuals = []
    if hasattr(engine, "generate_visuals"):
        try:
            visuals = engine.generate_visuals(df, visuals_dir)
        except Exception as e:
            log.warning(f"Visual generation failed for {domain}: {e}")

    # -------------------------------------------------
    # 4. RETURN STANDARDIZED PAYLOAD
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
        }
    }
