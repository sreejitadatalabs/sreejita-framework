import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape

# 🧠 EXECUTIVE COGNITION
from sreejita.narrative.executive_cognition import build_executive_payload

log = logging.getLogger("sreejita.orchestrator")


# =====================================================
# SAFE TABULAR LOADER
# =====================================================

def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        for enc in (None, "utf-8", "latin-1", "cp1252"):
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

    Responsibilities:
    - Load data
    - Detect dataset shape
    - Select domain
    - Run domain engine
    - Produce executive cognition payload

    Explicitly does NOT:
    - Interpret results
    - Rank KPIs
    - Format outputs
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. LOAD DATA (IMMUTABLE BASE)
    # -------------------------------------------------
    base_df = _read_tabular_file_safe(input_path)

    if base_df.empty:
        log.warning("Input file is empty: %s", input_path)

    # -------------------------------------------------
    # 2. DATASET SHAPE (CONTEXT ONLY)
    # -------------------------------------------------
    try:
        shape_info = detect_dataset_shape(base_df)
        log.info("Dataset shape detected: %s", shape_info.get("shape"))
    except Exception as e:
        log.exception("Shape detection failed")
        shape_info = {
            "shape": "unknown",
            "score": {},
            "signals": {}
        }

    # -------------------------------------------------
    # 3. DOMAIN DECISION
    # -------------------------------------------------
    decision = decide_domain(base_df)
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
                "executive": {},
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (DEFENSIVE & ISOLATED)
    # -------------------------------------------------
    try:
        # 🧠 ISOLATE DATAFRAME
        df = base_df.copy(deep=True)

        # Attach shape context to engine (optional, safe)
        engine.shape_info = shape_info
        engine.shape = shape_info.get("shape")

        # Preprocess
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        # KPIs (MANDATORY)
        kpis = engine.calculate_kpis(df) or {}

        # Insights (backward-safe)
        try:
            insights = engine.generate_insights(df, kpis, shape_info=shape_info) or []
        except TypeError:
            insights = engine.generate_insights(df, kpis) or []

        # Recommendations (backward-safe)
        try:
            raw_recs = engine.generate_recommendations(
                df, kpis, insights, shape_info=shape_info
            ) or []
        except TypeError:
            raw_recs = engine.generate_recommendations(df, kpis, insights) or []

        recommendations = enrich_recommendations(raw_recs) or []

        # 🧠 EXECUTIVE COGNITION (ALWAYS SAFE)
        executive_payload = build_executive_payload(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations
        )

    except Exception as e:
        log.exception("Domain processing failed: %s", domain)

        # ❗ DO NOT CRASH THE SYSTEM
        return {
            domain: {
                "kpis": {},
                "insights": [{
                    "level": "CRITICAL",
                    "title": "Domain Processing Failure",
                    "so_what": str(e),
                }],
                "recommendations": [],
                "visuals": [],
                "shape": shape_info,
                "executive": {},
            }
        }

    # -------------------------------------------------
    # 5. VISUAL GENERATION (HARDENED)
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
            if not isinstance(v, dict):
                continue

            path = v.get("path")
            if not path:
                continue

            p = Path(path)
            if not p.is_absolute():
                p = visuals_dir / p

            if p.exists():
                visuals.append({
                    **v,
                    "path": str(p)
                })

    # -------------------------------------------------
    # 6. AUTHORITATIVE PAYLOAD (CONTRACT SAFE)
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
            "shape": shape_info,

            # 🧠 EXECUTIVE DECISION INTELLIGENCE
            "executive": executive_payload,
        }
            }
