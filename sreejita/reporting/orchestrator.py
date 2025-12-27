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

    # CSV handling (encoding-safe)
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding="latin-1")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="cp1252")

    # Excel handling (engine-safe)
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
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH (v3.6.1)
# =====================================================

def generate_report_payload(input_path: str, config: dict) -> dict:
    """
    Orchestrator — SINGLE SOURCE OF TRUTH

    Responsibilities:
    - Load data safely (CSV + Excel)
    - Decide domain
    - Execute domain lifecycle
    - Generate visuals into run_dir/visuals
    - Enforce HARD visual guarantees
    - Return a STANDARDIZED payload
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if "run_dir" not in config:
        raise RuntimeError("config['run_dir'] is required")

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. LOAD DATA (SAFE)
    # -------------------------------------------------
    df = _read_tabular_file_safe(input_path)

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
    if hasattr(engine, "preprocess"):
        df = engine.preprocess(df)

    kpis = engine.calculate_kpis(df)
    insights = engine.generate_insights(df, kpis)

    raw_recommendations = engine.generate_recommendations(df, kpis)
    recommendations = enrich_recommendations(raw_recommendations)

    # -------------------------------------------------
    # 4. VISUALS (HARD GUARANTEE)
    # -------------------------------------------------
    visuals = []

    if hasattr(engine, "generate_visuals"):
        visuals_dir = run_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        try:
            visuals = engine.generate_visuals(df, visuals_dir)
        except Exception as e:
            raise RuntimeError(
                f"Visual generation failed for domain '{domain}': {e}"
            )

        # 🔒 HARD CONTRACT — visuals must exist on disk
        if visuals:
            for vis in visuals:
                if not isinstance(vis, dict):
                    raise RuntimeError("Visual entry must be a dict")

                path = vis.get("path")
                if not path:
                    raise RuntimeError("Visual entry missing 'path' key")

                p = Path(path)
                if not p.exists():
                    raise RuntimeError(
                        f"Visual declared but file not found: {p}"
                    )
        else:
            log.warning(
                "Domain '%s' returned no visuals (allowed but tracked)",
                domain,
            )

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
