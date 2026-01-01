import logging
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape

# 🧠 EXECUTIVE COGNITION
from sreejita.narrative.executive_cognition import build_executive_payload


# =====================================================
# LOGGER (DEFENSIVE)
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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
# BOARD READINESS PERSISTENCE (FILE-BASED)
# =====================================================

def _load_board_history(run_dir: Path) -> Dict[str, int]:
    path = run_dir / "board_readiness_history.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_board_history(run_dir: Path, history: Dict[str, int]) -> None:
    path = run_dir / "board_readiness_history.json"
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _compute_trend(previous: int, current: int) -> str:
    if previous is None:
        return "→"
    if current > previous + 2:
        return "↑"
    if current < previous - 2:
        return "↓"
    return "→"


# =====================================================
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH
# =====================================================

def generate_report_payload(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Authoritative orchestration layer.

    Responsibilities:
    - Load data
    - Detect dataset shape (context only)
    - Select domain
    - Execute domain engine
    - Generate executive cognition payload

    Explicitly does NOT:
    - Interpret KPIs
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
        shape_val = shape_info.get("shape")
        if hasattr(shape_val, "value"):
            shape_info["shape"] = shape_val.value

        log.info("Dataset shape detected: %s", shape_info.get("shape"))

    except Exception:
        log.exception("Shape detection failed")
        shape_info = {
            "shape": "unknown",
            "score": {},
            "signals": {},
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
                    "so_what": "No suitable domain engine could be identified.",
                }],
                "recommendations": [],
                "visuals": [],
                "shape": shape_info,
                "executive": {},
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (ISOLATED & DEFENSIVE)
    # -------------------------------------------------
    try:
        df = base_df.copy(deep=True)

        engine.shape_info = shape_info
        engine.shape = shape_info.get("shape")

        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        kpis = engine.calculate_kpis(df) or {}

        try:
            insights = engine.generate_insights(df, kpis, shape_info=shape_info) or []
        except TypeError:
            insights = engine.generate_insights(df, kpis) or []

        try:
            raw_recs = engine.generate_recommendations(
                df, kpis, insights, shape_info=shape_info
            ) or []
        except TypeError:
            raw_recs = engine.generate_recommendations(df, kpis, insights) or []

        recommendations = enrich_recommendations(raw_recs) or []

        # 🧠 EXECUTIVE COGNITION (PURE)
        executive_payload = build_executive_payload(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
        )

        # -------------------------------------------------
        # BOARD READINESS TREND (PERSISTENT)
        # -------------------------------------------------
        history = _load_board_history(run_dir)

        board = executive_payload.get("board_readiness", {})
        current_score = board.get("score")
        previous_score = history.get(domain)

        trend = _compute_trend(previous_score, current_score)

        executive_payload["board_readiness_trend"] = {
            "previous_score": previous_score,
            "current_score": current_score,
            "trend": trend,
        }

        if isinstance(current_score, int):
            history[domain] = current_score
            _save_board_history(run_dir, history)

    except Exception as e:
        log.exception("Domain processing failed: %s", domain)

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

            raw_path = v.get("path")
            if not raw_path:
                continue

            p = Path(raw_path)
            if not p.is_absolute():
                p = visuals_dir / p

            if p.exists():
                visuals.append({
                    **v,
                    "path": str(p),
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

            # 🧠 EXECUTIVE = SINGLE SOURCE OF TRUTH
            "executive": executive_payload,
        }
    }
