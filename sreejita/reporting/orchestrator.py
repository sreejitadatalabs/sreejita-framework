import logging
from pathlib import Path
import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape  # ✅ NEW

log = logging.getLogger("sreejita.orchestrator")


# =====================================================
# SAFE TABULAR LOADER (CSV + EXCEL)
# =====================================================

def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
    """
    Safely read CSV or Excel files with real-world fallbacks.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding="latin-1")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="cp1252")

    if suffix in (".xls", ".xlsx"):
        try:
            return pd.read_excel(path)
        except ValueError:
            try:
                return pd.read_excel(path, engine="openpyxl")
            except Exception:
                return pd.read_excel(path, engine="xlrd")

    raise ValueError(f"Unsupported file type: {suffix}")


# =====================================================
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH (SHAPE-AWARE)
# =====================================================

def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    Orchestrator — SINGLE SOURCE OF TRUTH

    GUARANTEES:
    - Dataset-shape aware
    - Domain-safe execution
    - Visuals never block
    - Narrative never breaks
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if "run_dir" not in config:
        raise RuntimeError("config['run_dir'] is required")

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------
    df = _read_tabular_file_safe(input_path)

    # -------------------------------------------------
    # 2. DATASET SHAPE DETECTION (🔥 CRITICAL)
    # -------------------------------------------------
    try:
        shape_info = detect_dataset_shape(df)
        log.info("Detected dataset shape: %s", shape_info.get("shape"))
    except Exception as e:
        log.warning("Dataset shape detection failed: %s", e)
        shape_info = {"shape": "unknown"}

    # -------------------------------------------------
    # 3. DOMAIN DECISION
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    if engine is None:
        log.warning("No engine found for domain: %s", domain)
        return {
            "unknown": {
                "kpis": {"rows": len(df), "columns": len(df.columns)},
                "insights": [{
                    "level": "RISK",
                    "title": "Unknown Domain",
                    "so_what": "No matching domain engine found.",
                }],
                "recommendations": [],
                "visuals": [],
                "shape": shape_info,
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN LIFECYCLE (SHAPE-AWARE)
    # -------------------------------------------------
    try:
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        kpis = engine.calculate_kpis(df)

        # 🔑 Shape-aware insights
        insights = engine.generate_insights(
            df,
            kpis,
            shape_info=shape_info
            if "shape_info" in engine.generate_insights.__code__.co_varnames
            else None
        )

        # 🔑 Shape-aware recommendations
        raw_recommendations = engine.generate_recommendations(
            df,
            kpis,
            insights,
            shape_info=shape_info
            if "shape_info" in engine.generate_recommendations.__code__.co_varnames
            else None
        )

        recommendations = enrich_recommendations(raw_recommendations)

    except Exception as e:
        log.exception("Domain engine failed: %s", domain)
        raise RuntimeError(f"Domain processing failed for '{domain}': {e}")

    # -------------------------------------------------
    # 5. VISUALS (FAIL-SAFE)
    # -------------------------------------------------
    visuals = []

    if hasattr(engine, "generate_visuals"):
        visuals_dir = run_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        try:
            generated = engine.generate_visuals(df, visuals_dir) or []
        except Exception as e:
            log.error(
                "Visual generation failed for domain '%s': %s",
                domain,
                e,
                exc_info=True,
            )
            generated = []

        for vis in generated:
            if (
                isinstance(vis, dict)
                and vis.get("path")
                and Path(vis["path"]).exists()
            ):
                visuals.append(vis)
            else:
                log.warning("Invalid visual skipped: %s", vis)

    # -------------------------------------------------
    # 6. STANDARDIZED PAYLOAD (AUTHORITATIVE)
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
            "shape": shape_info,  # ✅ EXPOSED FOR NARRATIVE & PDF
        }
    }
