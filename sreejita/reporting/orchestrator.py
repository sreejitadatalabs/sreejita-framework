import logging
from pathlib import Path
import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations

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
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH (STABLE)
# =====================================================

def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    Orchestrator — SINGLE SOURCE OF TRUTH

    GUARANTEES:
    - Never raises due to visuals
    - Always returns payload
    - Domain bugs never block PDF
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
    # 2. DOMAIN DECISION
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    if engine is None:
        log.warning("No engine found for domain: %s", domain)
        return {
            "unknown": {
                "kpis": {"rows": len(df), "columns": len(df.columns)},
                "insights": [
                    {
                        "level": "RISK",
                        "title": "Unknown Domain",
                        "so_what": "No matching domain engine found.",
                    }
                ],
                "recommendations": [],
                "visuals": [],
            }
        }

    # -------------------------------------------------
    # 3. DOMAIN LIFECYCLE
    # -------------------------------------------------
    try:
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        kpis = engine.calculate_kpis(df)
        insights = engine.generate_insights(df, kpis)

        raw_recommendations = engine.generate_recommendations(df, kpis)
        recommendations = enrich_recommendations(raw_recommendations)

    except Exception as e:
        # HARD FAIL ONLY FOR CORE ANALYTICS
        log.exception("Domain engine failed: %s", domain)
        raise RuntimeError(f"Domain processing failed for '{domain}': {e}")

    # -------------------------------------------------
    # 4. VISUALS (FAIL-SAFE — NEVER BLOCK)
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

        # Soft validation (NO RAISE)
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
    # 5. STANDARDIZED PAYLOAD (AUTHORITATIVE)
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
        }
    } 
