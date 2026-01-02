import logging
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape

# 🧠 EXECUTIVE COGNITION (MANDATORY)
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

    raise ValueError(f"Unsupported file type: {suffix}")


# =====================================================
# BOARD READINESS HISTORY (PERSISTENT)
# =====================================================

def _history_path(run_dir: Path) -> Path:
    return run_dir / "board_readiness_history.json"


def _load_history(run_dir: Path) -> Dict[str, int]:
    path = _history_path(run_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_history(run_dir: Path, history: Dict[str, int]) -> None:
    try:
        with open(_history_path(run_dir), "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _trend(prev: int | None, curr: int | None) -> str:
    if prev is None or curr is None:
        return "→"
    if curr >= prev + 5:
        return "↑"
    if curr <= prev - 5:
        return "↓"
    return "→"


# =====================================================
# ORCHESTRATOR — SINGLE SOURCE OF TRUTH
# =====================================================

def generate_report_payload(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if "run_dir" not in config:
        raise RuntimeError("config['run_dir'] is required")

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = input_path.stem

    # -------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------
    df = _read_tabular_file_safe(input_path)

    # -------------------------------------------------
    # 2. DATASET SHAPE (CONTEXT ONLY)
    # -------------------------------------------------
    try:
        shape_info = detect_dataset_shape(df)
    except Exception:
        shape_info = {"shape": "unknown", "signals": {}}

    # -------------------------------------------------
    # 3. DOMAIN DECISION
    # -------------------------------------------------
    decision = decide_domain(df)
    domain = decision.selected_domain
    engine = decision.engine

    if engine is None:
        return {
            "unknown": {
                "kpis": {},
                "insights": [{
                    "level": "RISK",
                    "title": "Unknown Domain",
                    "so_what": "No suitable domain engine found.",
                }],
                "recommendations": [],
                "visuals": [],
                "shape": shape_info,
                "executive": {},
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (DICT-BASED CONTRACT)
    # -------------------------------------------------
    try:
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        domain_result = engine.run(df)

        if not isinstance(domain_result, dict):
            raise RuntimeError(
                f"Domain engine '{domain}' returned invalid result type: "
                f"{type(domain_result)}"
            )

        kpis = domain_result.get("kpis", {})
        insights = domain_result.get("insights", [])
        raw_recs = domain_result.get("recommendations", [])
        visuals = domain_result.get("visuals", [])

        # Defensive recompute (preferred hooks)
        try:
            insights = engine.generate_insights(df, kpis, shape_info=shape_info)
        except TypeError:
            insights = engine.generate_insights(df, kpis)

        try:
            raw_recs = engine.generate_recommendations(
                df, kpis, insights, shape_info=shape_info
            )
        except TypeError:
            raw_recs = engine.generate_recommendations(df, kpis, insights)

        recommendations = enrich_recommendations(raw_recs)

        # 🧠 EXECUTIVE INTELLIGENCE (MANDATORY)
        executive = build_executive_payload(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
        )

    except Exception as e:
        log.exception("Domain processing failed")
        raise RuntimeError(str(e))

    # -------------------------------------------------
    # 5. BOARD READINESS TREND (PERSISTENT)
    # -------------------------------------------------
    history = _load_history(run_dir)

    board = executive.get("board_readiness", {})
    current_score = board.get("score")
    previous_score = history.get(dataset_key)

    trend = _trend(previous_score, current_score)

    history_values = list(history.values())
    if isinstance(current_score, int):
        history_values.append(current_score)

    executive["board_readiness_trend"] = {
        "previous_score": previous_score,
        "current_score": current_score,
        "trend": trend,
    }

    executive["board_readiness_history"] = history_values[-10:]

    if isinstance(current_score, int):
        history[dataset_key] = current_score
        _save_history(run_dir, history)

    # -------------------------------------------------
    # 6. VISUAL SAFETY & SORTING
    # -------------------------------------------------
    visuals = [
        v for v in visuals
        if isinstance(v, dict) and Path(v.get("path", "")).exists()
    ]
    visuals = sorted(
        visuals,
        key=lambda x: x.get("importance", 0),
        reverse=True
    )

    # -------------------------------------------------
    # 7. FINAL PAYLOAD (CANONICAL)
    # -------------------------------------------------
    return {
        domain: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "visuals": visuals,
            "shape": shape_info,
            "executive": executive,
        }
    }
