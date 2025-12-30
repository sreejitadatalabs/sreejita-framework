import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape

log = logging.getLogger("sreejita.orchestrator")


# =====================================================
# SAFE TABULAR LOADER
# =====================================================

def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        for enc in (None, "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue

    if suffix in (".xls", ".xlsx"):
        for engine in (None, "openpyxl", "xlrd"):
            try:
                return pd.read_excel(path, engine=engine)
            except Exception:
                continue

    raise ValueError(f"Unsupported or unreadable file: {path}")


# =====================================================
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH
# =====================================================

def generate_report_payload(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Authoritative orchestration layer.
    - No intelligence
    - No formatting
    - No assumptions
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------
    df = _read_tabular_file_safe(input_path)

    # -------------------------------------------------
    # 2. DATASET SHAPE (CONTEXT ONLY)
    # -------------------------------------------------
    try:
        shape_info = detect_dataset_shape(df)
        log.info("Dataset shape detected: %s", shape_info.get("shape"))
    except Exception as e:
        log.warning("Shape detection failed: %s", e)
        shape_info = {"shape": "unknown", "score": {}}

    # -------------------------------------------------
    # 3. DOMAIN DECISION
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    if not engine:
        log.warning("No domain engine resolved.")
        return {
            "unknown": {
                "kpis": {},
                "insights": [{
                    "level": "RISK",
                    "title": "Unknown Domain",
                    "so_what": "No suitable domain engine could be identified."
                }],
                "recommendations": [],
                "visuals": [],
                "shape": shape_info,
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (DEFENSIVE)
    # -------------------------------------------------
    try:
        # Preprocess
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        # KPIs (REQUIRED)
        kpis = engine.calculate_kpis(df) or {}

        # Insights (SHAPE-AWARE, SAFE)
        try:
            insights = engine.generate_insights(df, kpis, shape_info=shape_info) or []
        except TypeError:
            insights = engine.generate_insights(df, kpis) or []

        # Recommendations (SHAPE-AWARE, SAFE)
        try:
            raw_recs = engine.generate_recommendations(df, kpis, insights, shape_info=shape_info) or []
        except TypeError:
            raw_recs = engine.generate_recommendations(df, kpis, insights) or []

        recommendations = enrich_recommendations(raw_recs) or []

    except Exception as e:
        log.exception("Domain processing failed: %s", domain)
        raise RuntimeError(f"Domain processing failed for '{domain}': {e}")

    # -------------------------------------------------
    # 5. VISUAL GENERATION (FAIL-SAFE)
    # -------------------------------------------------
    visuals = []

    if hasattr(engine, "generate_visuals"):
        visuals_dir = run_dir / "visuals" / domain
        visuals_dir.mkdir(parents=True, exist_ok=True)

        try:
            generated = engine.generate_visuals(df, visuals_dir) or []
        except Exception as e:
            log.error("Visual generation failed for %s: %s", domain, e)
            generated = []

        for v in generated:
            if isinstance(v, dict) and v.get("path") and Path(v["path"]).exists():
                visuals.append(v)

    # -------------------------------------------------
    # 6. AUTHORITATIVE PAYLOAD (CONTRACT SAFE)
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis or {},
            "insights": insights or [],
            "recommendations": recommendations or [],
            "visuals": visuals or [],
            "shape": shape_info,
        }
    }
