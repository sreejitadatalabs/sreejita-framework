import logging
from pathlib import Path
import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations

log = logging.getLogger("sreejita.orchestrator")


def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
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
        except Exception:
            return pd.read_excel(path, engine="openpyxl")

    raise ValueError(f"Unsupported file type: {suffix}")


def generate_report_payload(input_path: str, config: dict) -> dict:
    input_path = Path(input_path)
    df = _read_tabular_file_safe(input_path)

    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    if engine is None:
        return {
            "unknown": {
                "kpis": {"rows": len(df), "cols": len(df.columns)},
                "insights": [{
                    "level": "RISK",
                    "title": "Unknown Domain",
                    "so_what": "No matching domain engine found.",
                }],
                "recommendations": [],
                "visuals": [],
            }
        }

    if hasattr(engine, "preprocess"):
        df = engine.preprocess(df)

    kpis = engine.calculate_kpis(df)
    insights = engine.generate_insights(df, kpis)
    recommendations = enrich_recommendations(
        engine.generate_recommendations(df, kpis)
    )

    visuals = []
    if hasattr(engine, "generate_visuals"):
        try:
            visuals_dir = Path(config["run_dir"]) / "visuals"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            visuals = engine.generate_visuals(df, visuals_dir)
        except Exception as e:
            log.warning("Visual generation failed: %s", e)

    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
        }
    }
