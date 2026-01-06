# =====================================================
# ORCHESTRATOR — UNIVERSAL (AUTHORITATIVE, LOCKED)
# Sreejita Framework v3.6 STABILIZED
# =====================================================

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape
from sreejita.core.fingerprint import dataframe_fingerprint

log = logging.getLogger("sreejita.orchestrator")

# =====================================================
# SAFE FILE LOADER (NEVER CRASH)
# =====================================================

def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for enc in (None, "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue

    if path.suffix.lower() in (".xls", ".xlsx"):
        for engine in (None, "openpyxl", "xlrd"):
            try:
                return pd.read_excel(path, engine=engine)
            except Exception:
                continue

    raise RuntimeError(f"Unsupported file type: {path.suffix}")

# =====================================================
# BOARD READINESS HISTORY (SAFE)
# =====================================================

def _history_path(run_dir: Path) -> Path:
    return run_dir / "board_readiness_history.json"


def _load_history(run_dir: Path) -> Dict[str, int]:
    try:
        with open(_history_path(run_dir), "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_history(run_dir: Path, history: Dict[str, int]) -> None:
    try:
        with open(_history_path(run_dir), "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _trend(prev: Optional[int], curr: Optional[int]) -> str:
    if prev is None or curr is None:
        return "→"
    if curr >= prev + 5:
        return "↑"
    if curr <= prev - 5:
        return "↓"
    return "→"

# =====================================================
# CANONICAL ENTRY POINT (ONLY ENTRY)
# =====================================================

def generate_report_payload(
    input_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    run_dir = Path(config.get("run_dir", "runs/current"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. LOAD DATA (IMMUTABLE SOURCE)
    # -------------------------------------------------
    raw_df = _read_tabular_file_safe(input_path)
    if raw_df.empty:
        raise RuntimeError("Dataset is empty")

    df = raw_df.copy(deep=False)
    dataset_key = dataframe_fingerprint(df)

    # -------------------------------------------------
    # 2. DATASET SHAPE (CONTEXT ONLY — NEVER DRIVES LOGIC)
    # -------------------------------------------------
    shape_info = detect_dataset_shape(df)

    # -------------------------------------------------
    # 3. DOMAIN DECISION (ENGINE ALWAYS ATTACHED)
    # -------------------------------------------------
    decision = decide_domain(df)
    engine = decision.engine
    domain = decision.selected_domain

    if engine is None or not isinstance(domain, str):
        raise RuntimeError("Domain resolution failed")

    # 🔒 CRITICAL: RESET DOMAIN STATE (NO LEAKAGE)
    if hasattr(engine, "_last_kpis"):
        engine._last_kpis = None

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (STRICT ORDER — NEVER CHANGE)
    # -------------------------------------------------
    try:
        # PREPROCESS (COPY SAFE)
        df = engine.preprocess(df)

        # KPIs — SINGLE SOURCE OF TRUTH
        kpis = engine.calculate_kpis(df)
        engine._last_kpis = kpis

        # VISUALS — MUST COME AFTER KPIs
        try:
            visuals = engine.generate_visuals(
                df=df,
                output_dir=run_dir / "visuals" / domain,
            ) or []
        except Exception:
            visuals = []

        # INSIGHTS — KPI BOUND
        insights = engine.generate_insights(df, kpis)

        # RECOMMENDATIONS — INSIGHT BOUND
        raw_recs = engine.generate_recommendations(df, kpis, insights)
        recommendations = enrich_recommendations(raw_recs)

        # EXECUTIVE COGNITION — LAST
        executive = engine.build_executive(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
        )

    except Exception as e:
        log.exception(
            f"Domain execution failed | domain={domain} | fingerprint={dataset_key}"
        )
        raise RuntimeError(f"{domain} execution failed: {e}")

    # -------------------------------------------------
    # 5. VISUAL VALIDATION (PATH + CONFIDENCE)
    # -------------------------------------------------
    valid_visuals: List[Dict[str, Any]] = []

    for v in visuals:
        try:
            path = Path(v.get("path", ""))
            conf = float(v.get("confidence", 0))
            if path.exists() and conf >= 0.30:
                valid_visuals.append(v)
        except Exception:
            continue

    # -------------------------------------------------
    # 6. UNIVERSAL VISUAL SAFETY NET (FINAL GUARD)
    # -------------------------------------------------
    valid_visuals = engine.ensure_minimum_visuals(
        valid_visuals,
        df,
        run_dir / "visuals" / domain,
    )

    # -------------------------------------------------
    # 7. EXECUTIVE SAFE SELECTION (MAX 6)
    # -------------------------------------------------
    valid_visuals = sorted(
        valid_visuals,
        key=lambda v: float(v.get("importance", 0)) * float(v.get("confidence", 1)),
        reverse=True,
    )[:6]

    # -------------------------------------------------
    # 8. BOARD READINESS TREND (NON-BLOCKING)
    # -------------------------------------------------
    history = _load_history(run_dir)

    board = executive.get("board_readiness", {}) if isinstance(executive, dict) else {}
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

    # -------------------------------------------------
    # 9. FINAL PAYLOAD (STRICT CONTRACT)
    # -------------------------------------------------
    return {
        domain: {
            "domain": domain,
            "kpis": kpis,
            "visuals": valid_visuals,
            "insights": insights,
            "recommendations": recommendations,
            "executive": executive,
            "shape": shape_info,
        }
    }
