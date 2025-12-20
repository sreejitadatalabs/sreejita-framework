from pathlib import Path
from datetime import datetime
import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations


def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    v3.1 ORCHESTRATOR — INTEGRATED LIFECYCLE
    
    Responsibilities:
    - Load & validate data
    - Route to correct domain engine
    - Execute analysis lifecycle (KPIs -> Insights -> Visuals)
    - Ensure visuals land in the correct run-specific folder
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
        raise ValueError(f"Data ingestion failed: {str(e)}")

    # -------------------------------------------------
    # 2. DOMAIN ROUTING
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    # Fallback for unknown data
    if engine is None:
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

    # D. Visuals (Path Management Fix)
    # Visuals must be saved where the report will live to ensure relative links work.
    # We use the config's output directory if provided, else a temp default.
    output_root = Path(config.get("output_dir", "reports"))
    
    # If orchestrator is called by CLI, this dir might be generic.
    # The CLI creates the timestamped folder, but the Orchestrator doesn't know it yet.
    # STRATEGY: Save visuals to `output_dir/visuals` and let the Report Engine link them.
    visuals_dir = output_root / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    visuals = []
    if hasattr(engine, "generate_visuals"):
        try:
            visuals = engine.generate_visuals(df, visuals_dir)
        except Exception as e:
            print(f"⚠️ Visual generation warning: {e}")

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
