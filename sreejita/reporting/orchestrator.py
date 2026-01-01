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

def _history_path(run_dir: Path) -> Path:
    return run_dir / "board_readiness_history.json"


def _load_board_history(run_dir: Path) -> Dict[str, int]:
    path = _history_path(run_dir)
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_board_history(run_dir: Path, history: Dict[str, int]) -> None:
    try:
        with open(_history_path(run_dir), "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _compute_trend(previous: int, current: int) -> str:
    if previous is None or current is None:
        return "→"
    if current >= previous + 5:
        return "↑"
    if current <= previous - 5:
        return "↓"
    return "→"


# =====================================================
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH
# =====================================================

def generate_report_payload(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = f"{input_path.stem}"

    # -------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------
    base_df = _read_tabular_file_safe(input_path)
    if base_df.empty:
        log.warning("Input file is empty: %s", input_path)

    # -------------------------------------------------
    # 2. DATASET SHAPE
    # -------------------------------------------------
    try:
        shape_info = detect_dataset_shape(base_df)
        shape_val = shape_info.get("shape")
        if hasattr(shape_val, "value"):
            shape_info["shape"] = shape_val.value
    except Exception:
        log.exception("Shape detection failed")
        shape_info = {"shape": "unknown", "score": {}, "signals": {}}

    # -------------------------------------------------
    # 3. DOMAIN DECISION
    # -------------------------------------------------
    decision = decide_domain(base_df)
    domain = decision.selected_domain
    engine = decision.engine

    if not engine:
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
    # 4. DOMAIN EXECUTION
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

        executive_payload = build_executive_payload(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
        )

        # -------------------------------------------------
        # BOARD READINESS TREND + ALERT
        # -------------------------------------------------
        history = _load_board_history(run_dir)

        board = executive_payload.get("board_readiness", {})
        current_score = board.get("score")
        previous_score = history.get(dataset_key)

        trend = _compute_trend(previous_score, current_score)

        executive_payload["board_readiness_trend"] = {
            "previous_score": previous_score,
            "current_score": current_score,
            "trend": trend,
        }

        # 🚨 Inject CRITICAL alert if major drop
        if (
            isinstance(previous_score, int)
            and isinstance(current_score, int)
            and previous_score - current_score >= 10
        ):
            insights.insert(0, {
                "level": "CRITICAL",
                "title": "Board Readiness Decline",
                "so_what": (
                    f"Board readiness dropped from {previous_score} to "
                    f"{current_score}. Executive escalation required."
                ),
                "source": "Executive Governance",
                "executive_summary_flag": True,
            })

        if isinstance(current_score, int):
            history[dataset_key] = current_score
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
    # 5. VISUAL GENERATION
    # -------------------------------------------------
    visuals = []
    if hasattr(engine, "generate_visuals"):
        visuals_dir = run_dir / "visuals" / domain
        visuals_dir.mkdir(parents=True, exist_ok=True)

        try:
            generated = engine.generate_visuals(df, visuals_dir) or []
        except Exception:
            generated = []

        for v in generated:
            p = Path(v.get("path", ""))
            if not p.is_absolute():
                p = visuals_dir / p
            if p.exists():
                visuals.append({**v, "path": str(p)})

    # -------------------------------------------------
    # 6. FINAL PAYLOAD
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
            "shape": shape_info,
            "executive": executive_payload,
        }
    }
