import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

from sreejita.core.column_resolver import resolve_column, resolve_semantics
from sreejita.core.dataset_shape import detect_dataset_shape
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# CONSTANTS
# =====================================================

MIN_SAMPLE_SIZE = 30

# NOTE:
# 'flag' is a generalized binary outcome proxy used across:
# mortality, alerts, specimen rejection, screening, immunization, etc.
# This is intentional for universal-domain support in v3.x.

# =====================================================
# SUB-DOMAINS (CANONICAL ENUM)
# =====================================================

class HealthcareSubDomain(str, Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    DIAGNOSTICS = "diagnostics"
    PHARMACY = "pharmacy"
    PUBLIC_HEALTH = "public_health"
    MIXED = "mixed"
    UNKNOWN = "unknown"

# =====================================================
# VISUAL INTELLIGENCE MAP (LOCKED CONTRACT)
# =====================================================

HEALTHCARE_VISUAL_MAP: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        "avg_los_trend",
        "bed_turnover",
        "readmission_risk",
        "discharge_hour",
        "acuity_vs_staffing",
        "ed_boarding",
        "mortality_trend",
    ],
    HealthcareSubDomain.CLINIC.value: [
        "no_show_by_day",
        "wait_time_split",
        "appointment_lag",
        "provider_utilization",
        "demographic_reach",
        "referral_funnel",
        "telehealth_mix",
    ],
    HealthcareSubDomain.DIAGNOSTICS.value: [
        "tat_percentiles",
        "critical_alert_time",
        "specimen_rejection",
        "device_downtime",
        "order_heatmap",
        "repeat_scan",
        "ordering_variance",
    ],
    HealthcareSubDomain.PHARMACY.value: [
        "spend_velocity",
        "refill_gap",
        "therapeutic_spend",
        "generic_rate",
        "prescribing_variance",
        "inventory_turn",
        "drug_alerts",
    ],
    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        "incidence_geo",
        "cohort_growth",
        "prevalence_age",
        "access_gap",
        "program_effect",
        "sdoh_overlay",
        "immunization_rate",
    ],
}

# =====================================================
# SAFE SIGNAL DETECTION (STRICT, DOMAIN-AWARE)
# =====================================================

def _has_signal(
    df: pd.DataFrame,
    col: Optional[str],
    min_coverage: float = 0.3,
) -> bool:
    """
    Column must exist AND meet minimum non-null coverage.
    """
    if df is None or df.empty:
        return False

    if not col or col not in df.columns:
        return False

    coverage = df[col].notna().mean()
    return coverage >= min_coverage

# =====================================================
# SUB-DOMAIN ELIGIBILITY CONTRACT (HARD GATE)
# These are REQUIRED columns, not boosters.
# =====================================================

SUBDOMAIN_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: ["date", "discharge_date", "los"],
    HealthcareSubDomain.CLINIC.value: ["doctor", "duration"],
    HealthcareSubDomain.DIAGNOSTICS.value: ["duration", "encounter"],
    HealthcareSubDomain.PHARMACY.value: ["fill_date", "supply"],
    HealthcareSubDomain.PUBLIC_HEALTH.value: ["population"],
}

def _eligible_subdomain(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    sub: str,
) -> bool:
    """
    A sub-domain is eligible ONLY if ALL required
    columns exist and have signal.
    """
    required = SUBDOMAIN_REQUIRED_COLUMNS.get(sub, [])
    if not required:
        return False

    return all(_has_signal(df, cols.get(col)) for col in required)

# =====================================================
# UNIVERSAL SUB-DOMAIN INFERENCE — HEALTHCARE (FINAL)
# =====================================================

def infer_healthcare_subdomains(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """
    Deterministic, evidence-gated healthcare sub-domain inference.

    Input MUST be resolved column map (not boolean signals).

    Returns:
    - {sub_domain: confidence}
    - Multiple entries indicate MIXED dataset
    """

    if not isinstance(cols, dict):
        return {HealthcareSubDomain.UNKNOWN.value: 1.0}

    scores: Dict[str, float] = {}

    # -------------------------------
    # HOSPITAL
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.HOSPITAL.value):
        signals = sum([
            int(_has_signal(df, cols.get("los"))),
            int(_has_signal(df, cols.get("bed_id"))),
            int(_has_signal(df, cols.get("admit_type"))),
            int(
                _has_signal(df, cols.get("date"))
                and _has_signal(df, cols.get("discharge_date"))
            ),
        ])
        scores[HealthcareSubDomain.HOSPITAL.value] = round(
            min(1.0, 0.35 + 0.15 * signals), 2
        )

    # -------------------------------
    # CLINIC
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.CLINIC.value):
        signals = sum([
            int(_has_signal(df, cols.get("doctor"))),
            int(_has_signal(df, cols.get("duration"))),
            int(_has_signal(df, cols.get("facility"))),
        ])
        scores[HealthcareSubDomain.CLINIC.value] = round(
            min(0.85, 0.30 + 0.15 * signals), 2
        )

    # -------------------------------
    # DIAGNOSTICS
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.DIAGNOSTICS.value):
        signals = sum([
            int(_has_signal(df, cols.get("duration"))),
            int(_has_signal(df, cols.get("encounter"))),
            int(_has_signal(df, cols.get("flag"))),
        ])
        scores[HealthcareSubDomain.DIAGNOSTICS.value] = round(
            min(0.85, 0.30 + 0.15 * signals), 2
        )

    # -------------------------------
    # PHARMACY (STRICT — HARD GATED)
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PHARMACY.value):
    
        # 🚫 HARD PHARMACY GATE (NON-NEGOTIABLE)
        if not (
            cols.get("fill_date")
            and cols.get("supply")
            and cols.get("cost")
            and cols.get("fill_date") in df.columns
            and cols.get("supply") in df.columns
            and cols.get("cost") in df.columns
        ):
            pass  # 🚫 DO NOT ACTIVATE PHARMACY
        else:
            signals = sum([
                int(_has_signal(df, cols.get("fill_date"))),
                int(_has_signal(df, cols.get("supply"))),
                int(_has_signal(df, cols.get("cost"))),
            ])
    
            scores[HealthcareSubDomain.PHARMACY.value] = round(
                min(0.80, 0.35 + 0.15 * signals), 2
            )

    # -------------------------------
    # PUBLIC HEALTH
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PUBLIC_HEALTH.value):
        signals = sum([
            int(_has_signal(df, cols.get("population"))),
            int(_has_signal(df, cols.get("flag"))),
        ])
        scores[HealthcareSubDomain.PUBLIC_HEALTH.value] = round(
            min(0.90, 0.40 + 0.20 * signals), 2
        )

    # -------------------------------
    # FINAL RESOLUTION
    # -------------------------------
    if not scores:
        return {HealthcareSubDomain.MIXED.value: 0.4}

    if len(scores) == 1:
        return scores

    strongest = max(scores.values())
    return {
        k: v for k, v in scores.items()
        if v >= max(0.45, strongest - 0.20)
    }   

# =====================================================
# HEALTHCARE DOMAIN (FIXED, SUBDOMAIN-SAFE)
# =====================================================
class HealthcareDomain(BaseDomain):
    name = "healthcare"

    # -------------------------------------------------
    # KPI ACCESS HELPERS (SAFE)
    # -------------------------------------------------
    @staticmethod
    def get_kpi(kpis: Dict[str, Any], sub: str, key: str):
        namespaced = f"{sub}_{key}"
        return kpis.get(namespaced, kpis.get(key))

    @staticmethod
    def get_kpi_confidence(kpis: Dict[str, Any], sub: str, key: str) -> float:
        conf_map = kpis.get("_confidence", {})
        return conf_map.get(f"{sub}_{key}", conf_map.get(key, 0.6))

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL + SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal Healthcare preprocessing.

        Guarantees:
        - Deterministic column semantics
        - No sub-domain leakage
        - No pharmacy hallucination
        - Mixed datasets supported
        """

        df = df.copy()
        self.shape_info = detect_dataset_shape(df)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (AUTHORITATIVE)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # ---------------- IDENTITY ----------------
            "pid": resolve_column(df, "patient_id"),

            # Encounter is optional and diagnostics-only
            "encounter": resolve_column(df, "encounter"),

            # ---------------- TIME ----------------
            "date": resolve_column(df, "admission_date"),
            "discharge_date": resolve_column(df, "discharge_date"),
            "fill_date": resolve_column(df, "fill_date"),

            # ---------------- DURATION ----------------
            "los": resolve_column(df, "length_of_stay"),
            "duration": resolve_column(df, "duration"),

            # ---------------- COST ----------------
            "cost": resolve_column(df, "cost"),

            # ---------------- FLAGS ----------------
            "readmitted": resolve_column(df, "readmitted"),
            "flag": resolve_column(df, "flag"),

            # ---------------- OPERATIONS ----------------
            "facility": resolve_column(df, "facility"),
            "doctor": resolve_column(df, "doctor"),
            "admit_type": resolve_column(df, "admission_type"),
            "bed_id": resolve_column(df, "bed_id"),

            # ---------------- PHARMACY / POPULATION ----------------
            "supply": resolve_column(df, "supply"),
            "population": resolve_column(df, "population"),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        for key in ("los", "duration", "cost", "supply", "population"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------------------------------
        # DATE NORMALIZATION
        # -------------------------------------------------
        for key in ("date", "discharge_date", "fill_date"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # -------------------------------------------------
        # BOOLEAN NORMALIZATION (LIMITED, NON-DESTRUCTIVE)
        # -------------------------------------------------
        BOOL_MAP = {
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
        }

        for key in ("readmitted", "flag"):
            col = self.cols.get(key)
            if col and col in df.columns:
                s = df[col].astype(str).str.lower().str.strip()
                mapped = s.map(BOOL_MAP)
                df[col] = pd.to_numeric(
                    mapped.where(mapped.notna(), df[col]),
                    errors="coerce",
                )

        # -------------------------------------------------
        # DERIVE LOS (ONLY IF SAFE)
        # -------------------------------------------------
        date_col = self.cols.get("date")
        discharge_col = self.cols.get("discharge_date")

        if (
            self.cols.get("los") is None
            and date_col
            and discharge_col
            and date_col in df.columns
            and discharge_col in df.columns
        ):
            delta = (df[discharge_col] - df[date_col]).dt.days
            delta = delta.where(delta.between(0, 365))

            df["__derived_los"] = pd.to_numeric(delta, errors="coerce")
            self.cols["los"] = "__derived_los"

        # -------------------------------------------------
        # CANONICAL TIME COLUMN (SAFE ORDER)
        # -------------------------------------------------
        self.time_col = None
        for key in ("date", "discharge_date", "fill_date"):
            col = self.cols.get(key)
            if col and col in df.columns:
                self.time_col = col
                break

        return df
   
    # -------------------------------------------------
    # KPI ENGINE (UNIVERSAL, SUB-DOMAIN HARD-LOCKED)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = int(len(df))
    
        # -------------------------------------------------
        # STEP 1: INFER SUB-DOMAINS (COLUMN-BASED, SAFE)
        # -------------------------------------------------
        inferred = infer_healthcare_subdomains(df, self.cols)
    
        if not inferred:
            active_subs = {}
            primary_sub = HealthcareSubDomain.MIXED.value
        elif HealthcareSubDomain.UNKNOWN.value in inferred:
            active_subs = {}
            primary_sub = HealthcareSubDomain.MIXED.value
            is_mixed = False
        else:
            ordered = sorted(inferred.items(), key=lambda x: x[1], reverse=True)
            primary_sub, primary_conf = ordered[0]
    
            active_subs = {primary_sub: primary_conf}
    
            for sub, conf in ordered[1:]:
                if conf >= 0.5 and abs(primary_conf - conf) <= 0.2:
                    active_subs[sub] = conf
    
            is_mixed = len(active_subs) > 1
    
        # -------------------------------------------------
        # STEP 2: BASE KPI CONTEXT
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "primary_sub_domain": (
                HealthcareSubDomain.MIXED.value if is_mixed else primary_sub
            ),
            "sub_domains": active_subs,
            "record_count": volume,
            "total_volume": volume,
            "data_completeness": round(1 - df.isna().mean().mean(), 3),
        }
    
        if (
            self.time_col
            and self.time_col in df.columns
            and df[self.time_col].notna().any()
        ):
            kpis["time_coverage_days"] = int(
                (df[self.time_col].max() - df[self.time_col].min()).days
            )
        else:
            kpis["time_coverage_days"] = None
    
        if volume < MIN_SAMPLE_SIZE:
            kpis["data_warning"] = "Sample size below recommended threshold"
    
        # -------------------------------------------------
        # SAFE KPI HELPERS (STRICT)
        # -------------------------------------------------
        def safe_mean(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        def safe_binary_rate(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            uniq = s.dropna().unique()
            if len(uniq) > 3:
                return None  # not binary-like
            return float((s > 0).mean()) if s.notna().any() else None
    
        # -------------------------------------------------
        # STEP 3: KPI COMPUTATION (SUB-DOMAIN HARD LOCKED)
        # -------------------------------------------------
        for sub, sub_conf in active_subs.items():
            prefix = f"{sub}_"
    
            # ---------------- HOSPITAL ----------------
            if sub == HealthcareSubDomain.HOSPITAL.value:
                los_col = self.cols.get("los")
                if los_col and los_col in df.columns:
                    avg_los = safe_mean(los_col)
                    kpis[f"{prefix}avg_los"] = avg_los
                    kpis[f"{prefix}long_stay_rate"] = (
                        float((df[los_col] > 7).mean())
                        if avg_los is not None
                        else None
                    )
    
                kpis[f"{prefix}readmission_rate"] = safe_binary_rate(
                    self.cols.get("readmitted")
                )
                kpis[f"{prefix}mortality_rate"] = safe_binary_rate(
                    self.cols.get("flag")
                )
                kpis[f"{prefix}er_boarding_time"] = safe_mean(
                    self.cols.get("duration")
                )
    
            # ---------------- CLINIC ----------------
            if sub == HealthcareSubDomain.CLINIC.value:
                doctor_col = self.cols.get("doctor")
                providers = (
                    df[doctor_col].nunique()
                    if doctor_col and doctor_col in df.columns
                    else None
                )
    
                kpis[f"{prefix}no_show_rate"] = safe_binary_rate(
                    self.cols.get("readmitted")
                )
                kpis[f"{prefix}avg_wait_time"] = safe_mean(
                    self.cols.get("duration")
                )
                kpis[f"{prefix}visit_cycle_time"] = safe_mean(
                    self.cols.get("duration")
                )
    
                if providers and providers > 0:
                    kpis[f"{prefix}visits_per_provider"] = volume / providers
    
            # ---------------- DIAGNOSTICS ----------------
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                kpis[f"{prefix}avg_tat"] = safe_mean(
                    self.cols.get("duration")
                )
                kpis[f"{prefix}specimen_rejection_rate"] = safe_binary_rate(
                    self.cols.get("flag")
                )
    
                doctor_col = self.cols.get("doctor")
                if doctor_col and doctor_col in df.columns:
                    staff = df[doctor_col].nunique()
                    if staff > 0:
                        kpis[f"{prefix}tests_per_fte"] = volume / staff
    
            # ---------------- PHARMACY (STRICT) ----------------
            if sub == HealthcareSubDomain.PHARMACY.value:
                fill_col = self.cols.get("fill_date")
                supply_col = self.cols.get("supply")
    
                if not (
                    fill_col and fill_col in df.columns
                    and supply_col and supply_col in df.columns
                ):
                    continue  # 🚫 NO PHARMACY KPIs
    
                kpis[f"{prefix}days_supply_on_hand"] = safe_mean(supply_col)
    
                cost_col = self.cols.get("cost")
                if cost_col and cost_col in df.columns:
                    kpis[f"{prefix}cost_per_rx"] = safe_mean(cost_col)
    
                kpis[f"{prefix}rx_volume"] = volume
    
            # ---------------- PUBLIC HEALTH ----------------
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                pop = safe_mean(self.cols.get("population"))
                cases_rate = safe_binary_rate(self.cols.get("flag"))
    
                if pop and cases_rate is not None:
                    kpis[f"{prefix}incidence_per_100k"] = min(cases_rate * 100_000, 100_000)

        kpis["_confidence"] = {}

        for k, v in kpis.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("_"):
                continue
        
            # Base confidence by data availability
            base = 0.6
        
            # Penalize small samples
            if volume < MIN_SAMPLE_SIZE:
                base -= 0.15
        
            # Penalize derived KPIs
            if "derived" in k or "proxy" in k:
                base -= 0.1
        
            # Bound confidence
            kpis["_confidence"][k] = round(
                max(0.35, min(0.85, base)),
                2,
            )

        kpis["_kpi_capabilities"] = {
            "avg_los": "time_flow",
            "long_stay_rate": "quality",
            "readmission_rate": "quality",
            "mortality_rate": "quality",
            "avg_wait_time": "time_flow",
            "avg_tat": "time_flow",
            "cost_per_rx": "cost",
            "incidence_per_100k": "quality",
            "record_count": "volume",
        }
        
        kpis["_domain_kpi_map"] = {
            sub: [k for k in kpis if k.startswith(f"{sub}_")]
            for sub in active_subs
        }
        # -------------------------------------------------
        # STEP 4: CACHE + RETURN
        # -------------------------------------------------
        self._last_kpis = kpis
        return kpis

    # -------------------------------------------------
    # VISUAL ENGINE (STRICT, KPI-LOCKED, EXECUTIVE SAFE)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
    
        output_dir.mkdir(parents=True, exist_ok=True)
    
        published: List[Dict[str, Any]] = []
        candidates: Dict[str, List[Dict[str, Any]]] = {}
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH: KPIs
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
        primary = kpis.get("primary_sub_domain")
    
        if not active_subs or primary == HealthcareSubDomain.UNKNOWN.value:
            return []
    
        visual_subs = (
            list(active_subs.keys())
            if primary == HealthcareSubDomain.MIXED.value
            else [primary]
        )
    
        # -------------------------------------------------
        # KPI EXISTENCE CHECK (CRITICAL FIX)
        # -------------------------------------------------
        def sub_has_any_kpi(sub: str) -> bool:
            prefix = f"{sub}_"
            return any(k.startswith(prefix) for k in kpis.keys())

        # -------------------------------------------------
        # SUB-DOMAIN CONFIDENCE WEIGHTING
        # -------------------------------------------------
        def sub_domain_weight(sub: str) -> float:
            return round(min(1.0, max(0.3, active_subs.get(sub, 0.3))), 2)
    
        # -------------------------------------------------
        # VISUAL REGISTRATION (CANDIDATE POOL)
        # -------------------------------------------------
        def register_visual(
            fig,
            name: str,
            caption: str,
            importance: float,
            base_confidence: float,
            sub_domain: str,
        ):
            fname = name if name.endswith(".png") else f"{name}.png"
            path = output_dir / fname
    
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    
            candidates.setdefault(sub_domain, []).append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "confidence": round(
                    min(0.95, base_confidence * sub_domain_weight(sub_domain)), 2
                ),
                "sub_domain": sub_domain,
                "inference_type": (
                    "derived" if "trend" in caption.lower()
                    else "proxy" if "proxy" in caption.lower()
                    else "direct"
                ),
            })
    
        # -------------------------------------------------
        # VISUAL DISPATCH (STRICTLY GUARDED)
        # -------------------------------------------------
        for sub in visual_subs:
    
            # 🚫 HARD KPI GATE — NO KPI, NO VISUALS
            if not sub_has_any_kpi(sub):
                continue
    
            # 🚫 PHARMACY HARD GATE
            if sub == HealthcareSubDomain.PHARMACY.value:
                if not (
                    self.cols.get("fill_date") in df.columns
                    and self.cols.get("supply") in df.columns
                ):
                    continue
    
            # 🚫 PUBLIC HEALTH HARD GATE
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                if self.cols.get("population") not in df.columns:
                    continue
    
            visual_keys = HEALTHCARE_VISUAL_MAP.get(sub, [])
            if not visual_keys:
                continue
    
            for visual_key in visual_keys:
                try:
                    self._render_visual_by_key(
                        visual_key=visual_key,
                        df=df.copy(deep=False),
                        output_dir=output_dir,
                        sub_domain=sub,
                        register_visual=register_visual,
                    )
                except Exception:
                    continue
    
        # -------------------------------------------------
        # FINAL SELECTION: MAX 6 VISUALS PER SUBDOMAIN
        # -------------------------------------------------
        for sub, pool in candidates.items():
            pool = [
                v for v in pool
                if Path(v["path"]).exists() and v["confidence"] >= 0.35
            ]
    
            pool.sort(
                key=lambda v: v["importance"] * v["confidence"],
                reverse=True,
            )
    
            published.extend(pool[:3])  # 🔒 HARD CAP
    
        return published

    # -------------------------------------------------
    # VISUAL RENDERER DISPATCH (REAL INTELLIGENCE)
    # -------------------------------------------------
    def _render_visual_by_key(
        self,
        visual_key: str,
        df: pd.DataFrame,
        output_dir: Path,
        sub_domain: str,
        register_visual,
    ):
        """
        Concrete visual implementations.
        Raises Exception if data is insufficient.
        """
        
        c = self.cols
        time_col = getattr(self, "time_col", None)
            
        # =================================================
        # SAFETY: minimum data requirement
        # =================================================
        if df is None or len(df) < 10:
            raise ValueError("Insufficient data for visualization")

        # Never mutate shared dataframe
        df = df.copy(deep=False)
        
        # =================================================
        # HOSPITAL VISUALS (STRICT & TIME-SAFE)
        # =================================================
        if sub_domain == "hospital":
            
            admit_col = c.get("date")
            discharge_col = c.get("discharge_date")
            los_col = c.get("los")
        
            # Ensure datetime safety where applicable
            for col in (admit_col, discharge_col):
                if col and col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        
            # -------------------------------------------------
            # 1. Average LOS Trend
            # -------------------------------------------------
            if visual_key == "avg_los_trend":
                time_axis = admit_col or discharge_col
                if not (time_axis and los_col):
                    raise ValueError("Time axis or LOS missing")
        
                series = (
                    df[[time_axis, los_col]]
                    .dropna()
                    .set_index(time_axis)[los_col]
                    .resample("M")
                    .mean()
                )
        
                if series.empty:
                    raise ValueError("No LOS data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                series.plot(ax=ax)
                ax.set_title("Average Length of Stay Trend", fontweight="bold")
                ax.set_ylabel("Days")
                ax.grid(alpha=0.3)
        
                register_visual(
                    fig,
                    "hospital_avg_los_trend.png",
                    "Monthly trend of inpatient length of stay.",
                    0.95,
                    0.90,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. Bed Turnover Velocity
            # -------------------------------------------------
            if visual_key == "bed_turnover":
                bed_col = c.get("bed_id")
                if not bed_col:
                    raise ValueError("Bed ID missing")
        
                counts = df[bed_col].dropna().value_counts()
                if counts.empty:
                    raise ValueError("No bed usage data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                counts.clip(upper=100).plot(kind="hist", bins=15, ax=ax)
                ax.set_title("Bed Turnover Velocity", fontweight="bold")
    
                register_visual(
                    fig,
                    "hospital_bed_velocity.png",
                    "Utilization frequency of hospital beds.",
                    0.92,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. Readmission Risk
            # -------------------------------------------------
            if visual_key == "readmission_risk":
                col = c.get("readmitted")
                if not col or col not in df.columns:
                    raise ValueError("Readmission column missing")
                    
                rates = df[col].dropna().value_counts(normalize=True)
                if rates.empty:
                    raise ValueError("No readmission data")
        
                rates.index = rates.index.map({0: "No", 1: "Yes"}).fillna(rates.index)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                rates.plot(kind="bar", ax=ax)
                ax.set_title("Readmission Rate Distribution", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "hospital_readmission.png",
                    "Distribution of 30-day readmissions.",
                    0.93,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. Discharge Hour Distribution
            # -------------------------------------------------
            if visual_key == "discharge_hour":
                if not discharge_col or discharge_col not in df.columns:
                    raise ValueError("Discharge date missing")
        
                hours = df[discharge_col].dt.hour.dropna()
                if hours.empty:
                    raise ValueError("No discharge hour data")
                    
                fig, ax = plt.subplots(figsize=(6, 4))
                hours.value_counts().sort_index().plot(kind="bar", ax=ax)
                ax.set_title("Discharge Hour Distribution", fontweight="bold")
                ax.set_xlabel("Hour of Day")
        
                register_visual(
                    fig,
                    "hospital_discharge_hour.png",
                    "Inpatient discharge timing pattern.",
                    0.85,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. Acuity vs Staffing (Proxy)
            # -------------------------------------------------
            if visual_key == "acuity_vs_staffing":
                if not (los_col and c.get("cost")):
                    raise ValueError("LOS or cost missing")
        
                cost_col = c.get("cost")
                if cost_col not in df.columns:
                    raise ValueError("Cost column missing")
        
                cost_cap = df[cost_col].quantile(0.95)
                if pd.isna(cost_cap):
                    raise ValueError("Invalid cost distribution")
        
                tmp = (
                    df[[los_col, cost_col]]
                    .dropna()
                    .clip(upper={los_col: 60, cost_col: cost_cap})
                )
        
                if tmp.empty:
                    raise ValueError("No acuity-cost data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(tmp[los_col], tmp[cost_col], alpha=0.35)
                ax.set_xlabel("Length of Stay (Acuity Proxy)")
                ax.set_ylabel("Cost (Staffing Proxy)")
                ax.set_title("Acuity vs Staffing Intensity (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "hospital_acuity_staffing.png",
                    "Relationship between patient acuity and staffing intensity (proxy).",
                    0.88,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. ED Boarding Time Trend
            # -------------------------------------------------
            if visual_key == "ed_boarding":
                dur_col = c.get("duration")
                if not (dur_col and admit_col):
                    raise ValueError("Duration or admission date missing")
        
                series = (
                    df[[admit_col, dur_col]]
                    .dropna()
                    .set_index(admit_col)[dur_col]
                    .resample("M")
                    .mean()
                )
        
                if series.empty:
                    raise ValueError("No ED boarding data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                series.plot(ax=ax)
                ax.set_title("ED Boarding Time Trend", fontweight="bold")
                ax.set_ylabel("Duration (units as recorded)")
        
                register_visual(
                    fig,
                    "hospital_ed_boarding.png",
                    "Average emergency department boarding time (proxy units).",
                    0.92,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. Mortality Trend (Proxy)
            # -------------------------------------------------
            if visual_key == "mortality_trend":
                flag_col = c.get("flag")
                if not (flag_col and admit_col):
                    raise ValueError("Mortality proxy or admission date missing")
        
                rate = (
                    df[[admit_col, flag_col]]
                    .dropna()
                    .set_index(admit_col)[flag_col]
                    .resample("M")
                    .mean()
                )
        
                if rate.empty:
                    raise ValueError("No mortality data")
    
                fig, ax = plt.subplots(figsize=(8, 4))
                rate.plot(ax=ax, marker="o")
                ax.set_title("In-Hospital Mortality Proxy Trend", fontweight="bold")
        
                register_visual(
                    fig,
                    "hospital_mortality_trend.png",
                    "Observed in-hospital mortality proxy trend over time.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
        
            raise ValueError(f"Unhandled hospital visual key: {visual_key}")

        # =================================================
        # CLINIC / AMBULATORY VISUALS (HARDENED & SAFE)
        # =================================================
        if sub_domain == "clinic":
        
            visit_col = c.get("date")
        
            if not visit_col or visit_col not in df.columns:
                raise ValueError("Clinic visit date missing")
        
            df = df.copy()
            df[visit_col] = pd.to_datetime(df[visit_col], errors="coerce")
        
            if df[visit_col].notna().sum() < 10:
                raise ValueError("Insufficient clinic visit records")
        
            # -------------------------------------------------
            # 1. NO-SHOW RATE BY DAY (PROXY)
            # -------------------------------------------------
            if visual_key == "no_show_by_day":
                flag_col = c.get("readmitted")
                if not flag_col or flag_col not in df.columns:
                    raise ValueError("No-show proxy column missing")
        
                tmp = df[[visit_col, flag_col]].dropna()
                if tmp.empty:
                    raise ValueError("No no-show data")
        
                tmp["_dow"] = tmp[visit_col].dt.day_name()
                rate = tmp.groupby("_dow")[flag_col].mean()
        
                day_order = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
                rate = rate.reindex(day_order).dropna()
        
                if rate.empty:
                    raise ValueError("No weekday no-show signal")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                rate.plot(kind="bar", ax=ax)
                ax.set_title("Appointment No-Show Rate by Day (Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "clinic_no_show_by_day.png",
                    "Appointment no-show rates across the week (proxy indicator).",
                    0.88,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. WAIT TIME TRAJECTORY
            # -------------------------------------------------
            if visual_key == "wait_time_split":
                dur_col = c.get("duration")
                if not dur_col or dur_col not in df.columns:
                    raise ValueError("Wait time column missing")
        
                series = (
                    df[[visit_col, dur_col]]
                    .dropna()
                    .assign(_dur=lambda x: x[dur_col].clip(upper=240))
                    .set_index(visit_col)["_dur"]
                    .resample("D")
                    .mean()
                )
        
                if series.empty:
                    raise ValueError("No wait-time trend data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                series.plot(ax=ax)
                ax.set_title("Average Patient Wait Time Trajectory", fontweight="bold")
                ax.set_ylabel("Minutes")
        
                register_visual(
                    fig,
                    "clinic_wait_time_trend.png",
                    "Trend of patient wait times from check-in to provider.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. APPOINTMENT LAG DISTRIBUTION (PROXY)
            # -------------------------------------------------
            if visual_key == "appointment_lag":
                pid_col = c.get("pid")
                if not pid_col or pid_col not in df.columns:
                    raise ValueError("Patient ID missing")
        
                tmp = df[[pid_col, visit_col]].dropna()
        
                lag = (
                    tmp.sort_values(visit_col)
                    .groupby(pid_col)[visit_col]
                    .diff()
                    .dt.days
                    .clip(lower=0, upper=60)
                    .dropna()
                )
        
                if lag.empty:
                    raise ValueError("No appointment lag data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                lag.hist(ax=ax, bins=20)
                ax.set_title("Appointment Lag Distribution (Proxy)", fontweight="bold")
                ax.set_xlabel("Days")
        
                register_visual(
                    fig,
                    "clinic_appointment_lag.png",
                    "Days between booking and clinic visit (proxy indicator).",
                    0.85,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. PROVIDER UTILIZATION
            # -------------------------------------------------
            if visual_key == "provider_utilization":
                doc_col = c.get("doctor")
                if not doc_col or doc_col not in df.columns:
                    raise ValueError("Doctor column missing")
        
                counts = (
                    df[doc_col]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                if counts.empty:
                    raise ValueError("No provider utilization data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                counts.plot(kind="bar", ax=ax)
                ax.set_title("Provider Utilization (Top 10)", fontweight="bold")
                ax.set_ylabel("Visits")
        
                register_visual(
                    fig,
                    "clinic_provider_utilization.png",
                    "Comparison of provider workload distribution.",
                    0.88,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PATIENT DEMOGRAPHIC REACH (LOCATION PROXY)
            # -------------------------------------------------
            if visual_key == "demographic_reach":
                fac_col = c.get("facility")
                if not fac_col or fac_col not in df.columns:
                    raise ValueError("Facility column missing")
        
                counts = df[fac_col].astype(str).value_counts()
                if counts.empty:
                    raise ValueError("No demographic reach data")
        
                top = counts.head(7)
                if len(counts) > 7:
                    top = pd.concat(
                        [top, pd.Series({"Other": counts.iloc[7:].sum()})]
                    )
        
                fig, ax = plt.subplots(figsize=(6, 6))
                top.plot(kind="pie", ax=ax, autopct="%1.0f%%")
                ax.set_ylabel("")
                ax.set_title("Patient Reach by Location (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "clinic_demographic_reach.png",
                    "Distribution of clinic visits by service location (proxy).",
                    0.82,
                    0.78,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. REFERRAL FUNNEL (ESTIMATED)
            # -------------------------------------------------
            if visual_key == "referral_funnel":
                total = len(df)
                if total < 20:
                    raise ValueError("Insufficient volume for referral funnel")
        
                stages = {
                    "Referrals": total,
                    "Scheduled (Est.)": int(total * 0.75),
                    "Completed (Est.)": int(total * 0.65),
                }
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(stages.keys(), stages.values())
                ax.set_title("Referral Conversion Funnel (Estimated)", fontweight="bold")
        
                register_visual(
                    fig,
                    "clinic_referral_funnel.png",
                    "Estimated referral flow from intake to completed visits.",
                    0.80,
                    0.70,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. TELEHEALTH MIX (PROXY)
            # -------------------------------------------------
            if visual_key == "telehealth_mix":
                fac_col = c.get("facility")
                if not fac_col or fac_col not in df.columns:
                    raise ValueError("Facility column missing")
        
                series = df[fac_col].astype(str).str.lower()
        
                mix = series.apply(
                    lambda x: "Telehealth" if "tele" in x else "In-Person"
                ).value_counts()
        
                if mix.empty:
                    raise ValueError("No telehealth mix data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                mix.plot(kind="bar", ax=ax)
                ax.set_title("Telehealth vs In-Person Visits (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "clinic_telehealth_mix.png",
                    "Service delivery mix across visit types (proxy indicator).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return
        
            raise ValueError(f"Unhandled clinic visual key: {visual_key}")

        # =================================================
        # DIAGNOSTICS (LABS / RADIOLOGY) VISUALS (HARDENED)
        # =================================================
        if sub_domain == "diagnostics":
        
            dur_col = c.get("duration")
            enc_col = c.get("encounter")
            flag_col = c.get("flag")
            doc_col = c.get("doctor")
            fac_col = c.get("facility")
        
            # -------------------------------
            # DIAGNOSTIC-SAFE TIME COLUMN
            # -------------------------------
            diag_time = (
                time_col if time_col in df.columns else None
            )
        
            if not diag_time:
                raise ValueError("Diagnostics requires a valid time column")
        
            df = df.copy()
            df[diag_time] = pd.to_datetime(df[diag_time], errors="coerce")
        
            if df[diag_time].notna().sum() < 15:
                raise ValueError("Insufficient diagnostic records")
        
            # -------------------------------------------------
            # 1. TURNAROUND TIME PERCENTILES (SMOOTHED)
            # -------------------------------------------------
            if visual_key == "tat_percentiles":
                if not dur_col or dur_col not in df.columns:
                    raise ValueError("Turnaround duration missing")
        
                tmp = (
                    df[[diag_time, dur_col]]
                    .dropna()
                    .assign(_dur=lambda x: x[dur_col].clip(upper=720))
                )
        
                if tmp.empty:
                    raise ValueError("No turnaround data")
        
                grouped = tmp.set_index(diag_time)["_dur"].resample("D")
                p50 = grouped.quantile(0.50).rolling(3, min_periods=1).mean()
                p90 = grouped.quantile(0.90).rolling(3, min_periods=1).mean()
                p95 = grouped.quantile(0.95).rolling(3, min_periods=1).mean()
        
                if p50.dropna().empty:
                    raise ValueError("Insufficient TAT distribution")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                p50.plot(ax=ax, label="50th %ile")
                p90.plot(ax=ax, label="90th %ile")
                p95.plot(ax=ax, label="95th %ile")
                ax.legend()
                ax.set_title("Diagnostic Turnaround Time Percentiles", fontweight="bold")
                ax.set_ylabel("Minutes")
        
                register_visual(
                    fig,
                    "diagnostics_tat_percentiles.png",
                    "Smoothed diagnostic turnaround time percentiles over time.",
                    0.92,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. CRITICAL ALERT NOTIFICATION TIME (PROXY)
            # -------------------------------------------------
            if visual_key == "critical_alert_time":
                if not (dur_col and flag_col):
                    raise ValueError("Duration or critical flag missing")
        
                dur = pd.to_numeric(df[dur_col], errors="coerce")
                flag = pd.to_numeric(df[flag_col], errors="coerce")
        
                critical = dur[flag == 1].clip(upper=180).dropna()
                if critical.empty:
                    raise ValueError("No critical alert timing data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                critical.hist(ax=ax, bins=20)
                ax.set_title("Critical Result Notification Time (Proxy)", fontweight="bold")
                ax.set_xlabel("Minutes")
        
                register_visual(
                    fig,
                    "diagnostics_critical_alert_time.png",
                    "Speed of notifying critical diagnostic results (proxy).",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. SPECIMEN REJECTION SIGNALS (PROXY)
            # -------------------------------------------------
            if visual_key == "specimen_rejection":
                if not flag_col:
                    raise ValueError("Specimen rejection flag missing")
        
                counts = (
                    pd.to_numeric(df[flag_col], errors="coerce")
                    .value_counts()
                    .head(5)
                )
        
                if counts.empty:
                    raise ValueError("No specimen rejection signals")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                counts.plot(kind="bar", ax=ax)
                ax.set_title("Specimen Rejection Signals (Proxy)", fontweight="bold")
                ax.set_ylabel("Count")
        
                register_visual(
                    fig,
                    "diagnostics_specimen_rejection.png",
                    "Observed specimen rejection indicators (proxy signals).",
                    0.88,
                    0.82,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. RELATIVE DEVICE UTILIZATION (FACILITY PROXY)
            # -------------------------------------------------
            if visual_key == "device_downtime":
                if not fac_col:
                    raise ValueError("Facility/device proxy missing")
        
                usage = (
                    df[fac_col]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                if usage.empty:
                    raise ValueError("No utilization proxy data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                usage.plot(kind="bar", ax=ax)
                ax.set_title("Relative Diagnostic Device Utilization (Proxy)", fontweight="bold")
                ax.set_ylabel("Observation Count")
        
                register_visual(
                    fig,
                    "diagnostics_device_utilization.png",
                    "Relative diagnostic equipment utilization by site (proxy).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PEAK ORDER LOAD HEATMAP
            # -------------------------------------------------
            if visual_key == "order_heatmap":
                tmp = df[[diag_time]].dropna()
                if tmp.empty:
                    raise ValueError("No diagnostic order timestamps")
        
                tmp["_hour"] = tmp[diag_time].dt.hour.clip(0, 23)
                tmp["_day"] = tmp[diag_time].dt.day_name()
        
                heat = pd.crosstab(tmp["_day"], tmp["_hour"])
        
                if heat.sum().sum() == 0:
                    raise ValueError("Empty diagnostic order heatmap")
        
                day_order = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
                heat = heat.reindex(day_order).fillna(0)
        
                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(heat, aspect="auto", cmap="Blues")
                ax.set_title("Peak Diagnostic Order Load", fontweight="bold")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Day of Week")
                plt.colorbar(im, ax=ax)
        
                register_visual(
                    fig,
                    "diagnostics_order_heatmap.png",
                    "Hourly diagnostic order intensity by day.",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. REPEAT SCAN INCIDENCE (ENCOUNTER-GUARDED)
            # -------------------------------------------------
            if visual_key == "repeat_scan":
                if not enc_col or enc_col not in df.columns:
                    raise ValueError("Encounter ID missing")
        
                counts = df[enc_col].value_counts()
                if counts[counts > 1].empty:
                    raise ValueError("No repeat diagnostic encounters")
        
                repeat_rate = (counts > 1).mean()
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Repeat Scan Rate"], [repeat_rate])
                ax.set_ylim(0, 1)
                ax.set_title("Repeat Diagnostic Incidence", fontweight="bold")
        
                register_visual(
                    fig,
                    "diagnostics_repeat_scan.png",
                    "Rate of repeated diagnostic tests indicating potential waste.",
                    0.86,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. PROVIDER ORDERING VARIANCE
            # -------------------------------------------------
            if visual_key == "ordering_variance":
                if not doc_col:
                    raise ValueError("Ordering provider missing")
        
                orders = (
                    df[doc_col]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                if orders.empty:
                    raise ValueError("No provider ordering variance")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                orders.plot(kind="bar", ax=ax)
                ax.set_title("Provider Ordering Variance", fontweight="bold")
                ax.set_ylabel("Orders")
        
                register_visual(
                    fig,
                    "diagnostics_ordering_variance.png",
                    "Variation in diagnostic ordering behavior across providers.",
                    0.87,
                    0.82,
                    sub_domain,
                )
                return
        
            raise ValueError(f"Unhandled diagnostics visual key: {visual_key}")

        # =================================================
        # PHARMACY VISUALS (STRICT & HARDENED)
        # =================================================
        if sub_domain == "pharmacy":
        
            fill_col = c.get("fill_date")
            supply_col = c.get("supply")
            cost_col = c.get("cost")
            pid_col = c.get("pid")
            doc_col = c.get("doctor")
            fac_col = c.get("facility")
            flag_col = c.get("flag")
        
            # -------------------------------------------------
            # HARD PHARMACY DATA GATE (NON-NEGOTIABLE)
            # -------------------------------------------------
            if not fill_col or fill_col not in df.columns:
                raise ValueError("Pharmacy requires prescription fill date")
        
            if not supply_col or supply_col not in df.columns:
                raise ValueError("Pharmacy requires days supply")
        
            if df[fill_col].notna().sum() < 10:
                raise ValueError("Insufficient pharmacy volume")
        
            df = df.copy()
            df[fill_col] = pd.to_datetime(df[fill_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. MEDICATION SPEND VELOCITY (CUMULATIVE, GUARDED)
            # -------------------------------------------------
            if visual_key == "spend_velocity":
                if not cost_col:
                    raise ValueError("Cost column missing")
        
                tmp = (
                    df[[fill_col, cost_col]]
                    .dropna()
                    .assign(_cost=lambda x: pd.to_numeric(x[cost_col], errors="coerce"))
                    .dropna()
                )
        
                if tmp.empty or tmp[fill_col].nunique() < 2:
                    raise ValueError("Insufficient spend density")
        
                spend = (
                    tmp.set_index(fill_col)["_cost"]
                    .resample("M")
                    .sum()
                    .cumsum()
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                spend.plot(ax=ax)
                ax.set_title("Medication Spend Velocity", fontweight="bold")
                ax.set_ylabel("Cumulative Spend")
        
                ax.yaxis.set_major_formatter(
                    lambda x, _: f"{x/1_000_000:.1f}M"
                    if x >= 1_000_000 else f"{x/1_000:.0f}K"
                )
        
                register_visual(
                    fig,
                    "pharmacy_spend_velocity.png",
                    "Cumulative medication expenditure over time.",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. REFILL ADHERENCE GAP (PATIENT-SAFE, PROXY)
            # -------------------------------------------------
            if visual_key == "refill_gap":

                tmp = df[[fill_col, supply_col]].dropna()
                if pid_col and pid_col in df.columns:
                    tmp[pid_col] = df[pid_col]
    
                if len(tmp) < 2:
                    raise ValueError("Insufficient refill history")
    
                # 🔴 FIX: enforce sort before shift
                if pid_col and pid_col in tmp.columns:
                    tmp = tmp.sort_values([pid_col, fill_col])
                    expected = (
                        tmp.groupby(pid_col)[fill_col].shift()
                        + pd.to_timedelta(
                            tmp.groupby(pid_col)[supply_col].shift(), unit="D"
                        )
                    )
                    gap = (tmp[fill_col] - expected).dt.days.dropna()
                else:
                    tmp = tmp.sort_values(fill_col)
                    expected = tmp[fill_col] + pd.to_timedelta(tmp[supply_col], unit="D")
                    gap = (tmp[fill_col].shift(-1) - expected).dt.days.dropna()
    
                gap = gap.clip(-30, 60)
                if gap.empty:
                    raise ValueError("No refill gap signal")
    
                fig, ax = plt.subplots(figsize=(6, 4))
                gap.hist(ax=ax, bins=20)
                ax.set_title("Refill Adherence Gap (Proxy)", fontweight="bold")
    
                register_visual(
                    fig,
                    "pharmacy_refill_gap.png",
                    "Delay between expected and actual refills (proxy).",
                    0.88,
                    0.78,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. THERAPEUTIC SPEND DISTRIBUTION (FACILITY PROXY)
            # -------------------------------------------------
            if visual_key == "therapeutic_spend":
                if not (fac_col and cost_col):
                    raise ValueError("Facility or cost missing")
        
                spend = (
                    df[[fac_col, cost_col]]
                    .dropna()
                    .assign(_cost=lambda x: pd.to_numeric(x[cost_col], errors="coerce"))
                    .groupby(fac_col)["_cost"]
                    .sum()
                    .nlargest(6)
                )
        
                if spend.empty:
                    raise ValueError("No therapeutic spend data")
        
                fig, ax = plt.subplots(figsize=(6, 6))
                spend.plot(kind="pie", autopct="%1.0f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Medication Spend Distribution (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "pharmacy_therapeutic_spend.png",
                    "Medication spend distribution by proxy grouping.",
                    0.82,
                    0.72,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. GENERIC SUBSTITUTION RATE (VERY WEAK PROXY)
            # -------------------------------------------------
            if visual_key == "generic_rate":
                if not fac_col:
                    raise ValueError("Facility proxy missing")
        
                series = (
                    df[fac_col]
                    .astype(str)
                    .str.lower()
                    .str.contains("generic", na=False)
                )
        
                rate = series.mean()
                if pd.isna(rate):
                    raise ValueError("Invalid generic proxy")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Generic Substitution Rate"], [rate])
                ax.set_ylim(0, 1)
                ax.set_title("Generic Substitution Rate (Weak Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "pharmacy_generic_rate.png",
                    "Estimated share of generic substitutions (weak proxy).",
                    0.70,
                    0.60,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PRESCRIBING COST VARIANCE (PROVIDER-LEVEL)
            # -------------------------------------------------
            if visual_key == "prescribing_variance":
                if not (doc_col and cost_col):
                    raise ValueError("Doctor or cost missing")
        
                variance = (
                    df[[doc_col, cost_col]]
                    .dropna()
                    .assign(_cost=lambda x: pd.to_numeric(x[cost_col], errors="coerce"))
                    .groupby(doc_col)["_cost"]
                    .mean()
                    .nlargest(10)
                )
        
                if variance.empty:
                    raise ValueError("No prescribing variance")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                variance.plot(kind="bar", ax=ax)
                ax.set_title("Prescribing Cost Variance", fontweight="bold")
                ax.set_ylabel("Average Cost")
        
                register_visual(
                    fig,
                    "pharmacy_prescribing_variance.png",
                    "Variation in average prescribing cost across providers.",
                    0.86,
                    0.82,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. INVENTORY TURNOVER (PROXY, GUARDED)
            # -------------------------------------------------
            if visual_key == "inventory_turn":
                supply = pd.to_numeric(df[supply_col], errors="coerce")
                cost = pd.to_numeric(df[cost_col], errors="coerce") if cost_col else None
        
                if supply.dropna().empty or cost is None:
                    raise ValueError("Inventory data insufficient")
        
                avg_supply = supply.mean()
                if avg_supply < 1:
                    raise ValueError("Supply baseline too small for turnover")

                turn = cost.sum() / avg_supply
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Inventory Turn Ratio"], [turn])
                ax.set_title("Inventory Turn Ratio (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "pharmacy_inventory_turn.png",
                    "Efficiency of medication inventory turnover (proxy).",
                    0.78,
                    0.70,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. DRUG SAFETY INTERVENTIONS (PROXY)
            # -------------------------------------------------
            if visual_key == "drug_alerts":
                if not flag_col:
                    raise ValueError("Safety alert flag missing")
        
                alerts = (
                    pd.to_numeric(df[flag_col], errors="coerce")
                    .value_counts(normalize=True)
                    .sort_index()
                )
        
                if alerts.empty:
                    raise ValueError("No drug safety alerts")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                alerts.plot(kind="bar", ax=ax)
                ax.set_title("Pharmacist Safety Interventions (Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "pharmacy_drug_alerts.png",
                    "Frequency of pharmacist safety interventions (proxy).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return
        
            raise ValueError(f"Unhandled pharmacy visual key: {visual_key}")

        # =================================================
        # PUBLIC HEALTH / POPULATION HEALTH VISUALS (HARDENED)
        # =================================================
        if sub_domain == "public_health":
        
            pop_col = c.get("population")
            flag_col = c.get("flag")
            fac_col = c.get("facility")
            pid_col = c.get("pid")
        
            # -------------------------------------------------
            # HARD PUBLIC HEALTH DATA GATE (NON-NEGOTIABLE)
            # -------------------------------------------------
            if not pop_col or pop_col not in df.columns:
                raise ValueError("Population column required")
        
            if not flag_col or flag_col not in df.columns:
                raise ValueError("Outcome flag required")
        
            pop = pd.to_numeric(df[pop_col], errors="coerce").dropna()
            flag = pd.to_numeric(df[flag_col], errors="coerce").fillna(0)
        
            if pop.empty or flag.sum() < 5:
                raise ValueError("Insufficient public health signal")
        
            pop_denom = pop.median()
            if pop_denom <= 0:
                raise ValueError("Invalid population denominator")
        
            # Ensure datetime safety once
            if time_col and time_col in df.columns:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. DISEASE INCIDENCE RATE (PER 100K, PROXY)
            # -------------------------------------------------
            if visual_key == "incidence_geo":
        
                incidence_rate = min((flag.mean()) * 100_000, 100_000)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Incidence per 100k"], [incidence_rate])
                ax.set_title("Disease Incidence Rate (Proxy)", fontweight="bold")
                ax.set_ylabel("Cases per 100,000")
        
                register_visual(
                    fig,
                    "public_health_incidence_rate.png",
                    "Estimated disease incidence per 100,000 population (proxy).",
                    0.90,
                    0.82,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. COHORT GROWTH TRAJECTORY (EVENT-BASED)
            # -------------------------------------------------
            if visual_key == "cohort_growth":
                if not time_col:
                    raise ValueError("Time column required")
        
                tmp = df[[time_col, flag_col]].dropna()
                if tmp.empty:
                    raise ValueError("No cohort data")
        
                cohort = (
                    (tmp[flag_col] == 1)
                    .astype(int)
                    .groupby(tmp[time_col].dt.to_period("M"))
                    .sum()
                    .cumsum()
                    .rolling(2, min_periods=1)
                    .mean()
                )
        
                if cohort.empty:
                    raise ValueError("Invalid cohort growth")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                cohort.plot(ax=ax)
                ax.set_title("Cohort Growth Trajectory (Smoothed)", fontweight="bold")
                ax.set_ylabel("Cumulative Events")
        
                register_visual(
                    fig,
                    "public_health_cohort_growth.png",
                    "Smoothed cumulative growth of observed public health events.",
                    0.88,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. DEMOGRAPHIC SEGMENT PREVALENCE (VERY WEAK PROXY)
            # -------------------------------------------------
            if visual_key == "prevalence_age":
                if not pid_col:
                    raise ValueError("Patient identifier required")
        
                pid_len = df[pid_col].astype(str).str.len()
                buckets = pd.cut(
                    pid_len,
                    bins=[0, 6, 8, 10, 99],
                    labels=["Segment A", "Segment B", "Segment C", "Segment D"],
                )
        
                prevalence = (
                    flag.groupby(buckets)
                    .mean()
                    .dropna()
                )
        
                if prevalence.empty:
                    raise ValueError("No demographic prevalence signal")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                prevalence.plot(kind="bar", ax=ax)
                ax.set_title("Outcome Prevalence by Demographic Segment (Weak Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "public_health_demographic_prevalence.png",
                    "Outcome prevalence by inferred demographic segments (weak proxy).",
                    0.75,
                    0.65,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. SERVICE ACCESS INDICATOR (PROXY)
            # -------------------------------------------------
            if visual_key == "access_gap":
                if not fac_col:
                    raise ValueError("Facility column required")
        
                service_points = df[fac_col].astype(str).nunique()
        
                ratio = (service_points / pop_denom) * 1_000
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Service Points per 1k"], [ratio])
                ax.set_title("Healthcare Access Indicator (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "public_health_access_gap.png",
                    "Availability of healthcare service points per 1,000 residents (proxy).",
                    0.85,
                    0.78,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PROGRAM EFFECTIVENESS TREND (PROXY)
            # -------------------------------------------------
            if visual_key == "program_effect":
                if not time_col:
                    raise ValueError("Time column required")
        
                tmp = df[[time_col, flag_col]].dropna()
                if tmp.empty:
                    raise ValueError("No program data")
        
                trend = (
                    (tmp[flag_col] == 1)
                    .astype(int)
                    .groupby(tmp[time_col].dt.to_period("M"))
                    .mean()
                    .rolling(3, min_periods=1)
                    .mean()
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                trend.plot(ax=ax)
                ax.set_title("Program Outcome Trend (Proxy)", fontweight="bold")
                ax.set_ylabel("Outcome Rate")
        
                register_visual(
                    fig,
                    "public_health_program_effect.png",
                    "Smoothed public health outcome trends following interventions (proxy).",
                    0.86,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. SOCIAL DETERMINANTS RISK OVERLAY (PROXY)
            # -------------------------------------------------
            if visual_key == "sdoh_overlay":
                if not fac_col:
                    raise ValueError("Facility column required")
        
                sdoh = (
                    df[[fac_col, flag_col]]
                    .dropna()
                    .groupby(fac_col)[flag_col]
                    .mean()
                    .nlargest(8)
                )
        
                if sdoh.empty:
                    raise ValueError("No SDOH signal")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                sdoh.plot(kind="bar", ax=ax)
                ax.set_title("Social Determinants Risk Overlay (Proxy)", fontweight="bold")
                ax.set_ylabel("Outcome Rate")
        
                register_visual(
                    fig,
                    "public_health_sdoh_overlay.png",
                    "Health outcome variation across socioeconomic regions (proxy).",
                    0.80,
                    0.72,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. IMMUNIZATION / SCREENING COVERAGE (PROXY)
            # -------------------------------------------------
            if visual_key == "immunization_rate":
                rate = flag.mean()
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Coverage Rate"], [rate])
                ax.set_ylim(0, 1)
                ax.set_title("Immunization / Screening Coverage (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "public_health_immunization_rate.png",
                    "Estimated population coverage of immunization or screening programs (proxy).",
                    0.84,
                    0.76,
                    sub_domain,
                )
                return
        
            raise ValueError(f"Unhandled public health visual key: {visual_key}")

    # -------------------------------------------------
    # INSIGHTS ENGINE (EVIDENCE-LOCKED, EXECUTIVE SAFE)
    # -------------------------------------------------
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:
    
        insights: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # CONFIDENCE CALCULATION (HONEST, BOUNDED)
        # -------------------------------------------------
        def insight_conf(kpi_conf: float, sub_score: float) -> float:
            base = min(kpi_conf or 0.0, 0.85)
            return round(min(0.92, base * (0.6 + 0.4 * sub_score)), 2)
    
        # -------------------------------------------------
        # NORMALIZE SUB-DOMAIN KEYS (SAFETY)
        # -------------------------------------------------
        HOSP = HealthcareSubDomain.HOSPITAL.value
        CLIN = HealthcareSubDomain.CLINIC.value
        DIAG = HealthcareSubDomain.DIAGNOSTICS.value
        PHAR = HealthcareSubDomain.PHARMACY.value
        PUBH = HealthcareSubDomain.PUBLIC_HEALTH.value
    
        # -------------------------------------------------
        # CROSS-DOMAIN INSIGHT (STRICTLY GUARDED)
        # -------------------------------------------------
        hosp_score = active_subs.get(HOSP, 0.0)
        diag_score = active_subs.get(DIAG, 0.0)
    
        if hosp_score >= 0.5 and diag_score >= 0.5:
            los = self.get_kpi(kpis, HOSP, "avg_los")
            tat = self.get_kpi(kpis, DIAG, "avg_tat")
    
            if isinstance(los, (int, float)) and isinstance(tat, (int, float)) and tat > 120:
                conf_val = min(
                    self.get_kpi_confidence(kpis, HOSP, "avg_los"),
                    self.get_kpi_confidence(kpis, DIAG, "avg_tat"),
                )
    
                insights.append({
                    "sub_domain": "cross_domain",
                    "level": "RISK",
                    "title": "Diagnostic Turnaround Delays Linked to Inpatient Stay",
                    "so_what": (
                        f"Extended diagnostic turnaround times "
                        f"({tat:.0f} minutes) are likely contributing to longer "
                        f"inpatient stays (average LOS {los:.1f} days), "
                        "indicating a cross-functional throughput constraint."
                    ),
                    "confidence": insight_conf(conf_val, min(hosp_score, diag_score)),
                })
    
        # -------------------------------------------------
        # SUB-DOMAIN INSIGHTS (MAX 5 EACH, NO FILLERS)
        # -------------------------------------------------
        for sub, score in active_subs.items():
            sub_insights: List[Dict[str, Any]] = []
    
            # -------------------------------
            # STRENGTHS
            # -------------------------------
            if sub == HOSP:
                if score < 0.6:
                    continue
                    
                avg_los = self.get_kpi(kpis, sub, "avg_los")
                if isinstance(avg_los, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Inpatient Throughput Visibility",
                        "so_what": (
                            f"Length of stay is consistently observable "
                            f"(average {avg_los:.1f} days), enabling effective "
                            "inpatient throughput governance."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_los"),
                            score,
                        ),
                    })
    
            if sub == CLIN:
                if score < 0.6:
                    continue
                    
                wait = self.get_kpi(kpis, sub, "avg_wait_time")
                if isinstance(wait, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Clinic Access Transparency",
                        "so_what": (
                            f"Patient wait times are measurable "
                            f"(average {wait:.0f} minutes), supporting "
                            "access and scheduling optimization."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_wait_time"),
                            score,
                        ),
                    })
    
            if sub == DIAG:
                if score < 0.6:
                    continue
                    
                tat = self.get_kpi(kpis, sub, "avg_tat")
                if isinstance(tat, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Diagnostic Turnaround Observability",
                        "so_what": (
                            f"Turnaround times are tracked "
                            f"(average {tat:.0f} minutes), enabling SLA monitoring "
                            "and operational accountability."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_tat"),
                            score,
                        ),
                    })
    
            if sub == PHAR:
                if score < 0.6:
                    continue
                    
                cost = self.get_kpi(kpis, sub, "cost_per_rx")
                if isinstance(cost, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Prescription Cost Visibility",
                        "so_what": (
                            f"Average prescription cost is observable "
                            f"(₹{cost:.0f}), supporting pharmacy spend oversight."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "cost_per_rx"),
                            score,
                        ),
                    })
    
            if sub == PUBH:
                if score < 0.6:
                    continue
                    
                inc = self.get_kpi(kpis, sub, "incidence_per_100k")
                if isinstance(inc, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Population Health Signal Coverage",
                        "so_what": (
                            "Population-level indicators enable monitoring "
                            "of disease burden and prevention effectiveness."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "incidence_per_100k"),
                            score,
                        ),
                    })
    
            # -------------------------------
            # WARNINGS / RISKS
            # -------------------------------
            if sub == CLIN:
                no_show = self.get_kpi(kpis, sub, "no_show_rate")
                if isinstance(no_show, (int, float)) and no_show >= 0.10:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Elevated Appointment No-Show Risk",
                        "so_what": (
                            f"A no-show rate of {no_show:.1%} may reduce "
                            "clinic throughput and revenue efficiency."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "no_show_rate"),
                            score,
                        ),
                    })
    
            if sub == HOSP:
                long_stay = self.get_kpi(kpis, sub, "long_stay_rate")
                if isinstance(long_stay, (int, float)) and long_stay >= 0.25:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Discharge Throughput Constraint",
                        "so_what": (
                            f"{long_stay:.1%} of patients exceed expected "
                            "length-of-stay thresholds, indicating discharge bottlenecks."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "long_stay_rate"),
                            score,
                        ),
                    })
    
            if sub == DIAG:
                tat = self.get_kpi(kpis, sub, "avg_tat")
                if isinstance(tat, (int, float)) and tat > 120:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Diagnostic Turnaround Pressure",
                        "so_what": (
                            f"Average turnaround time of {tat:.0f} minutes "
                            "may delay clinical decision-making."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_tat"),
                            score,
                        ),
                    })
    
            if sub == PUBH:
                inc = self.get_kpi(kpis, sub, "incidence_per_100k")
                kpi_conf = self.get_kpi_confidence(kpis, sub, "incidence_per_100k")
                if (
                    isinstance(inc, (int, float))
                    and inc > 300
                    and kpi_conf >= 0.5
                ):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Elevated Disease Incidence",
                        "so_what": (
                            f"Incidence rate of {inc:.0f} per 100k exceeds "
                            "expected thresholds, indicating prevention gaps."
                        ),
                        "confidence": insight_conf(kpi_conf, score),
                    })
    
            insights.extend(sub_insights[:5])
    
        # -------------------------------------------------
        # EXECUTIVE-STABLE SORTING
        # -------------------------------------------------
        level_order = {"RISK": 0, "WARNING": 1, "STRENGTH": 2}
        insights.sort(
            key=lambda x: (
                level_order.get(x["level"], 3),
                -x["confidence"],
            )
        )
    
        return insights

    # --------------------------------
    # RECOMMENDATIONS ENGINE (INSIGHT-BOUND, EXECUTIVE SAFE)
    # --------------------------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *_,
    ) -> List[Dict[str, Any]]:
    
        recommendations: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # INDEX INSIGHTS BY SUB-DOMAIN
        # -------------------------------------------------
        insights_by_sub: Dict[str, List[Dict[str, Any]]] = {}
        for ins in insights:
            if isinstance(ins, dict):
                insights_by_sub.setdefault(ins.get("sub_domain"), []).append(ins)
    
        # -------------------------------------------------
        # CONFIDENCE BINDING (NON-INFLATING)
        # -------------------------------------------------
        def rec_conf(ins_conf: float, sub_score: float) -> float:
            base = min(ins_conf, 0.85)
            return round(min(0.90, base * (0.7 + 0.3 * sub_score)), 2)
    
        # -------------------------------------------------
        # NORMALIZED SUB KEYS
        # -------------------------------------------------
        HOSP = HealthcareSubDomain.HOSPITAL.value
        CLIN = HealthcareSubDomain.CLINIC.value
        DIAG = HealthcareSubDomain.DIAGNOSTICS.value
        PHAR = HealthcareSubDomain.PHARMACY.value
        PUBH = HealthcareSubDomain.PUBLIC_HEALTH.value
    
        # -------------------------------------------------
        # SUB-DOMAIN RECOMMENDATIONS (ISOLATED)
        # -------------------------------------------------
        for sub, score in active_subs.items():
    
            sub_insights = insights_by_sub.get(sub, [])
            if not sub_insights:
                continue  # 🔒 no insight → no recommendation
    
            sub_recs: List[Dict[str, Any]] = []
    
            for ins in sub_insights:
                level = ins.get("level")
                ins_conf = float(ins.get("confidence", 0.6))
    
                # ===============================
                # HOSPITAL
                # ===============================
                if sub == HOSP and level == "RISK":
                    los = self.get_kpi(kpis, sub, "avg_los")
    
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Establish centralized discharge coordination and daily LOS review",
                        "owner": "Hospital Operations",
                        "timeline": "30–60 days",
                        "goal": "Reduce prolonged inpatient stays and free bed capacity",
                        "current_value": round(los, 2) if isinstance(los, (int, float)) else None,
                        "expected_impact": "10–15% improvement in bed availability",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # CLINIC
                # ===============================
                if sub == CLIN and level == "WARNING":
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "MEDIUM",
                        "action": "Deploy predictive no-show mitigation and automated reminders",
                        "owner": "Ambulatory Operations",
                        "timeline": "30–90 days",
                        "goal": "Stabilize provider utilization and improve access",
                        "expected_impact": "5–10% increase in completed visits",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # DIAGNOSTICS
                # ===============================
                if sub == DIAG and level == "RISK":
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Implement STAT workflow escalation and turnaround time governance",
                        "owner": "Diagnostics Leadership",
                        "timeline": "30–60 days",
                        "goal": "Reduce diagnostic turnaround delays",
                        "expected_impact": "Faster clinical decision-making",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # PHARMACY
                # ===============================
                if sub == PHAR and level == "WARNING":
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "MEDIUM",
                        "action": "Strengthen pharmacist safety review and alert resolution workflows",
                        "owner": "Pharmacy Operations",
                        "timeline": "30–60 days",
                        "goal": "Reduce medication safety risk",
                        "expected_impact": "Lower intervention and error rates",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # PUBLIC HEALTH (CONFIDENCE-GATED)
                # ===============================
                if sub == PUBH and level == "RISK" and score >= 0.6:
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Target high-incidence regions with focused prevention programs",
                        "owner": "Public Health Authority",
                        "timeline": "90–180 days",
                        "goal": "Reduce disease incidence and population risk",
                        "expected_impact": "Lower future healthcare burden",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
            # -------------------------------
            # DEDUP + PER-SUB CAP (MAX 2)
            # -------------------------------
            seen = set()
            deduped = []
            for r in sub_recs:
                key = r["action"]
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
    
            recommendations.extend(deduped[:2])
    
        # -------------------------------------------------
        # EXECUTIVE SORTING & HARD CAP
        # -------------------------------------------------
        priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(
            key=lambda r: (
                priority_rank.get(r.get("priority"), 3),
                -r.get("confidence", 0),
            )
        )
    
        # BOARD SAFE
        return recommendations[:8]

# =====================================================
# HEALTHCARE DOMAIN DETECTOR (ALIAS + COVERAGE AWARE)
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    semantics = resolve_semantics(df)

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        """
        Universal, capability-based healthcare domain detector.

        Guarantees:
        - Alias-aware (via column_resolver)
        - Pharmacy / clinic / hospital safe
        - Never drops domain once anchored
        - Confidence reflects strength, not existence
        """

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={},
            )

        # -------------------------------------------------
        # SEMANTIC CAPABILITIES (AUTHORITATIVE)
        # -------------------------------------------------
        semantics = resolve_semantics(df)

        # -------------------------------------------------
        # HEALTHCARE DOMAIN ANCHOR (CAPABILITY-BASED)
        # -------------------------------------------------
        anchor_score = sum([
            semantics.get("has_patient_id", False),
            semantics.get("has_admission_date", False),
            semantics.get("has_discharge_date", False),
            semantics.get("has_duration", False),   # clinic / diagnostics
            semantics.get("has_cost", False),       # billing / pharmacy
            semantics.get("has_supply", False),     # pharmacy
            semantics.get("has_population", False), # public health
        ])

        # ❗ If no healthcare signal at all → not healthcare
        if anchor_score == 0:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals=semantics,
            )

        # -------------------------------------------------
        # CONFIDENCE SCORING (LINEAR, HONEST)
        # -------------------------------------------------
        # Base confidence once healthcare is anchored
        confidence = 0.30

        # Each additional capability strengthens confidence
        confidence += min(anchor_score * 0.10, 0.55)

        confidence = round(min(confidence, 0.95), 2)

        # -------------------------------------------------
        # RETURN — NEVER DROP DOMAIN AFTER ANCHOR
        # -------------------------------------------------
        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals=semantics,
        )

# -----------------------------------------------------
# REGISTRY HOOK
# -----------------------------------------------------
def register(registry):
    registry.register(
        "healthcare",
        HealthcareDomain,
        HealthcareDomainDetector,
    )
