# =====================================================
# ORCHESTRATOR — UNIVERSAL (FINAL)
# Sreejita Framework
# =====================================================

import logging
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape

# 🧠 Executive cognition (single source of truth)
from sreejita.narrative.executive_cognition import build_executive_payload

log = logging.getLogger("sreejita.orchestrator")


# =====================================================
# SAFE TABULAR FILE LOADER
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
# ORCHESTRATOR ENTRY — CANONICAL
# =====================================================

def generate_report_payload(
    input_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:

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

    # -------------------------------------------------
    # 3A. UNKNOWN DOMAIN — GOVERNED FALLBACK
    # -------------------------------------------------
    if engine is None:
        executive = build_executive_payload(
            kpis={
                "primary_sub_domain": "unknown",
                "total_volume": len(df),
                "data_completeness": round(1 - df.isna().mean().mean(), 3),
                "_confidence": {},
                "_kpi_capabilities": {},
            },
            insights=[{
                "level": "RISK",
                "title": "Unknown or Unsupported Domain",
                "so_what": (
                    "The dataset does not match any supported domain patterns. "
                    "Only governance-level assessment is possible."
                ),
                "confidence": 0.6,
            }],
            recommendations=[{
                "priority": "HIGH",
                "action": "Review dataset structure and domain relevance",
                "owner": "Data Governance",
                "timeline": "Immediate",
                "goal": "Enable domain-specific analysis",
                "confidence": 0.7,
            }],
        )

        return {
            "unknown": {
                "kpis": {},
                "insights": executive["insights"],
                "recommendations": executive["recommendations"],
                "visuals": [],
                "shape": shape_info,
                "executive": executive,
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (STRICT CONTRACT)
    # -------------------------------------------------
    try:
        if hasattr(engine, "preprocess"):
            df = engine.preprocess(df)

        kpis = engine.calculate_kpis(df)

        visuals = engine.generate_visuals(
            df=df,
            output_dir=run_dir / "visuals" / domain
        )

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

    except Exception as e:
        log.exception("Domain processing failed")
        raise RuntimeError(str(e))

    # -------------------------------------------------
    # 5. VISUAL SAFETY ENFORCEMENT (NON-NEGOTIABLE)
    # -------------------------------------------------
    visuals = [
        v for v in visuals
        if isinstance(v, dict) and Path(v.get("path", "")).exists()
    ]

    visuals = sorted(
        visuals,
        key=lambda x: (x.get("importance", 0) * x.get("confidence", 1)),
        reverse=True,
    )

    evidence_summary = {
        "visual_count": len(visuals),
        "has_min_visuals": len(visuals) >= 2,
    }

    # -------------------------------------------------
    # 6. EXECUTIVE COGNITION (FINAL)
    # -------------------------------------------------
    executive = build_executive_payload(
        kpis=kpis,
        insights=insights,
        recommendations=recommendations,
    )

    executive["evidence_summary"] = evidence_summary

    # -------------------------------------------------
    # 7. BOARD READINESS TREND (PERSISTENT)
    # -------------------------------------------------
    history = _load_history(run_dir)

    board = executive.get("board_readiness", {})
    current_score = board.get("score")
    previous_score = history.get(dataset_key)

    executive["board_readiness_trend"] = {
        "previous_score": previous_score,
        "current_score": current_score,
        "trend": _trend(previous_score, current_score),
    }

    if isinstance(current_score, int):
        history[dataset_key] = current_score
        _save_history(run_dir, history)

    executive["board_readiness_history"] = list(history.values())[-10:]

    # -------------------------------------------------
    # 8. FINAL PAYLOAD (CANONICAL)
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
