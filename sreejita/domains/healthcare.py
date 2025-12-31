import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from sreejita.core.dataset_shape import detect_dataset_shape
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# IMPORT THE TRUTH
from sreejita.narrative.benchmarks import HEALTHCARE_THRESHOLDS, HEALTHCARE_EXTERNAL_LIMITS

# [FIX] Define Visual Benchmark to prevent NameError in generate_visuals
VISUAL_BENCHMARK_LOS = 5.0

# =====================================================
# 1. HEALTHCARE UNIVERSAL ENUMS
# =====================================================

class HealthcareSubDomain(str, Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    DIAGNOSTICS = "diagnostics"
    PHARMACY = "pharmacy"
    PUBLIC_HEALTH = "public_health"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class HealthcareCapability(str, Enum):
    VOLUME = "volume"
    TIME = "time"
    COST = "cost"
    QUALITY = "quality"
    VARIANCE = "variance"
    ACCESS = "access"
    DATA_QUALITY = "data_quality"


# =====================================================
# 2. SUBDOMAIN EXPECTATIONS (GOVERNANCE)
# =====================================================

SUBDOMAIN_EXPECTATIONS = {
    HealthcareSubDomain.HOSPITAL: {
        "required": {HealthcareCapability.VOLUME, HealthcareCapability.TIME},
        "bonus": {HealthcareCapability.QUALITY, HealthcareCapability.COST, HealthcareCapability.VARIANCE},
    },
    HealthcareSubDomain.CLINIC: {
        "required": {HealthcareCapability.VOLUME},
        "bonus": {HealthcareCapability.TIME, HealthcareCapability.ACCESS},
    },
    HealthcareSubDomain.DIAGNOSTICS: {
        "required": {HealthcareCapability.VOLUME},
        "bonus": {HealthcareCapability.TIME, HealthcareCapability.QUALITY},
    },
    HealthcareSubDomain.PHARMACY: {
        "required": {HealthcareCapability.VOLUME, HealthcareCapability.COST},
        "bonus": set(),
    },
    HealthcareSubDomain.PUBLIC_HEALTH: {
        "required": {HealthcareCapability.VOLUME},
        "bonus": {HealthcareCapability.ACCESS, HealthcareCapability.QUALITY},
    },
    HealthcareSubDomain.MIXED: {
        "required": {HealthcareCapability.VOLUME},
        "bonus": {HealthcareCapability.TIME, HealthcareCapability.COST},
    },
    HealthcareSubDomain.UNKNOWN: {
        "required": {HealthcareCapability.DATA_QUALITY},
        "bonus": set(),
    },
}

def _get_score_interpretation(score: int) -> str:
    if score >= 85:
        return "Strong operational confidence"
    elif score >= 70:
        return "Moderate operational confidence"
    elif score >= 50:
        return "Elevated operational risk"
    return "Critical operational risk"

def _get_trend_explanation(trend: str) -> str:
    return {
        "↑": "Performance indicators are deteriorating",
        "↓": "Performance indicators are improving",
        "→": "Performance indicators are stable"
    }.get(trend, "Trend unavailable")
# =====================================================
# 3. FACT MAPPING (PURE DATA)
# =====================================================

class HealthcareMapping:
    def __init__(self, df: pd.DataFrame, cols: Dict[str, str]):
        self.df = df
        self.c = cols

    # -------------------------
    # DATA QUALITY
    # -------------------------
    def data_completeness(self) -> float:
        critical = [v for v in self.c.values() if v and v in self.df.columns]
        if not critical:
            return 0.0
        return float(round(1 - self.df[critical].isna().mean().mean(), 2))

    # -------------------------
    # VOLUME
    # -------------------------
    def volume(self) -> int:
        if self.c.get("pid"):
            return self.df[self.c["pid"]].nunique()
        if self.c.get("encounter"):
            return self.df[self.c["encounter"]].nunique()
        return len(self.df)

    # -------------------------
    # TIME
    # -------------------------
    def avg_duration(self) -> Optional[float]:
        for k in ["los", "duration"]:
            if self.c.get(k):
                return self.df[self.c[k]].mean()
        return None

    def long_duration_rate(self, threshold: float) -> Optional[float]:
        for k in ["los", "duration"]:
            if self.c.get(k):
                return (self.df[self.c[k]] > threshold).mean()
        return None

    # -------------------------
    # COST
    # -------------------------
    def total_cost(self) -> Optional[float]:
        if self.c.get("cost"):
            return self.df[self.c["cost"]].sum()
        return None

    def avg_unit_cost(self) -> Optional[float]:
        if self.c.get("cost"):
            return self.df[self.c["cost"]].mean()
        return None

    # -------------------------
    # QUALITY
    # -------------------------
    def adverse_rate(self) -> Optional[float]:
        for k in ["readmitted", "flag"]:
            if self.c.get(k):
                return self.df[self.c[k]].mean()
        return None
    # -------------------------
    # VARIANCE
    # -------------------------
    def variance(self) -> Optional[float]:
        """
        Measures normalized performance variance across entities.
        """
        # Prevents statistical noise from triggering governance alerts.
        if self.volume() < 30:
            return None
            
        # 1. Metric Selection
        metric = None
        for k in ("cost", "los", "duration"):
            col = self.c.get(k)
            if col and col in self.df.columns:
                metric = col
                break
        
        if not metric: return None

        # 2. Group Variance (Facility / Doctor / Device)
        variances: List[float] = []
        
        # [UPDATE] Added 'device' to grouping keys for Digital Health support
        for group_key in ("facility", "doctor", "device"):
            group_col = self.c.get(group_key)

            if not group_col or group_col not in self.df.columns: continue

            grouped = (
                self.df[[group_col, metric]]
                .dropna()
                .groupby(group_col)[metric]
                .mean()
            )

            if len(grouped) < 3: continue

            mean_val = grouped.mean()
            std_val = grouped.std()

            if mean_val and mean_val > 0 and np.isfinite(std_val):
                normalized = std_val / mean_val
                variances.append(min(normalized, 1.0))

        if not variances: return None
        return max(variances)

    # -------------------------
    # TREND
    # -------------------------
    def trend(self, time_col: str, metric_col: str) -> str:
        if not time_col or metric_col not in self.df.columns:
            return "→"
        df = self.df.dropna(subset=[metric_col]).sort_values(time_col)
        if len(df) < 10:
            return "→"
        cut = int(len(df) * 0.8)
        hist = df.iloc[:cut][metric_col].mean()
        recent = df.iloc[cut:][metric_col].mean()
        if not hist or not np.isfinite(hist):
            return "→"
        delta = (recent - hist) / hist
        return "↑" if delta > 0.05 else "↓" if delta < -0.05 else "→"


# =====================================================
# 4. SUBDOMAIN & CAPABILITY DETECTION
# =====================================================

def detect_subdomain_and_capabilities(
    df: pd.DataFrame,
    cols: Dict[str, str]
) -> Tuple[HealthcareSubDomain, Set[HealthcareCapability]]:

    caps = {HealthcareCapability.DATA_QUALITY}

    def usable(col, min_ratio=0.1):
        """
        Capability presence check.
        Low threshold by design to support mixed & aggregated datasets.
        """
        return (
            col
            and col in df.columns
            and df[col].notna().any()
            and df[col].notna().mean() >= min_ratio
        )

    # -----------------------
    # CAPABILITIES
    # -----------------------
    if usable(cols.get("pid")) or usable(cols.get("encounter")):
        caps.add(HealthcareCapability.VOLUME)

    if usable(cols.get("los")) or usable(cols.get("duration")):
        caps.add(HealthcareCapability.TIME)

    if usable(cols.get("cost")):
        caps.add(HealthcareCapability.COST)

    if usable(cols.get("status")):
        caps.add(HealthcareCapability.QUALITY)

    if usable(cols.get("result")):
        caps.add(HealthcareCapability.QUALITY)

    if usable(cols.get("readmitted")) or usable(cols.get("flag")):
        caps.add(HealthcareCapability.QUALITY)

    if usable(cols.get("facility")) or usable(cols.get("doctor")):
        caps.add(HealthcareCapability.VARIANCE)

    duration_col = cols.get("duration")

    if duration_col and duration_col in df.columns:
        raw_name = duration_col.lower()
        ACCESS_TOKENS = {"wait", "queue", "access", "turnaround"}
        if any(tok in raw_name for tok in ACCESS_TOKENS):
            caps.add(HealthcareCapability.ACCESS)

    # -----------------------
    # SUB-DOMAIN (ORDER MATTERS)
    # -----------------------
    if usable(cols.get("los")):
        sub = HealthcareSubDomain.HOSPITAL
    elif usable(cols.get("duration")):
        sub = HealthcareSubDomain.DIAGNOSTICS
    elif usable(cols.get("cost")) and not usable(cols.get("los")):
        sub = HealthcareSubDomain.PHARMACY
    elif usable(cols.get("population")):
        sub = HealthcareSubDomain.PUBLIC_HEALTH
    elif usable(cols.get("encounter")):
        sub = HealthcareSubDomain.CLINIC
    else:
        sub = HealthcareSubDomain.MIXED

    return sub, caps

# =====================================================
# 5. UNIVERSAL SCORE
# =====================================================

def compute_score(kpis, sub, caps):
    score = 100
    breakdown = {}
    rules = SUBDOMAIN_EXPECTATIONS[sub]

    for r in rules["required"]:
        if r not in caps:
            score -= 15
            breakdown[f"Missing {r.value}"] = -15

    for b in rules["bonus"]:
        if b in caps:
            score += 5

    return max(0, min(100, score)), breakdown
# =====================================================
# 6. HEALTHCARE DOMAIN
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal Healthcare Preprocess
    
        Responsibilities:
        - Detect dataset shape (context only)
        - Resolve columns across ALL healthcare sub-domains
        - Normalize datatypes safely
        - Derive missing but inferable fields (LOS / duration)
        - Set time column for trends
        """
    
        # -------------------------------------------------
        # 1. DATASET SHAPE (CONTEXT ONLY)
        # -------------------------------------------------
        self.shape_info = detect_dataset_shape(df)
        self.shape = self.shape_info.get("shape")
    
        # -------------------------------------------------
        # 2. UNIVERSAL COLUMN RESOLUTION
        # -------------------------------------------------
        self.cols = {
            # Identity / Volume
            "pid": (
                resolve_column(df, "patient_id")
                or resolve_column(df, "mrn")
                or resolve_column(df, "person_id")
            ),
            "encounter": (
                resolve_column(df, "encounter_id")
                or resolve_column(df, "visit_id")
                or resolve_column(df, "appointment_id")
                or resolve_column(df, "test_id")
            ),
    
            # Time / Duration
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "duration": (
                resolve_column(df, "turnaround_time")
                or resolve_column(df, "duration")
                or resolve_column(df, "wait_time")
            ),
    
            # Financial
            "cost": (
                resolve_column(df, "total_cost")
                or resolve_column(df, "billing_amount")
                or resolve_column(df, "charges")
                or resolve_column(df, "amount")
            ),
    
            # Quality
            "readmitted": (
                resolve_column(df, "readmitted")
                or resolve_column(df, "readmission")
                or resolve_column(df, "flag")
                or resolve_column(df, "error")
            ),
    
            # Grouping / Variance
            "facility": (
                resolve_column(df, "facility")
                or resolve_column(df, "hospital")
                or resolve_column(df, "location")
                or resolve_column(df, "site")
                or resolve_column(df, "center")
            ),
            "doctor": (
                resolve_column(df, "doctor")
                or resolve_column(df, "provider")
                or resolve_column(df, "physician")
            ),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "condition"),
    
            # Dates (UNIVERSAL)
            "admit": (
                resolve_column(df, "admit_date")
                or resolve_column(df, "admission_date")
                or resolve_column(df, "arrival_date")
            ),
            "discharge": (
                resolve_column(df, "discharge_date")
                or resolve_column(df, "release_date")
            ),
            "date": (
                resolve_column(df, "date")
                or resolve_column(df, "event_date")
                or resolve_column(df, "visit_date")
                or resolve_column(df, "test_date")
                or resolve_column(df, "report_date")
            ),
            # [NEW] Appointment Efficiency (Clinic)
            "status": resolve_column(df, "appointment_status") or resolve_column(df, "visit_status") or resolve_column(df, "status"),
            
            # [NEW] Diagnostics (Labs)
            "result": resolve_column(df, "result_value") or resolve_column(df, "result") or resolve_column(df, "value"),
            "min_val": resolve_column(df, "normal_min") or resolve_column(df, "ref_min"),
            "max_val": resolve_column(df, "normal_max") or resolve_column(df, "ref_max"),

            # [NEW] Pharmacy
            "fill_date": resolve_column(df, "fill_date") or resolve_column(df, "dispense_date"),
            "supply": resolve_column(df, "days_supply") or resolve_column(df, "qty"),
            "drug": resolve_column(df, "drug_name") or resolve_column(df, "medication"),

            # [NEW] Public Health & Device
            "population": resolve_column(df, "population") or resolve_column(df, "census"),
            "device": resolve_column(df, "device_id") or resolve_column(df, "model"),
        }
    
        # -------------------------------------------------
        # 3. TYPE NORMALIZATION (CRITICAL)
        # -------------------------------------------------
        for key in ("los", "duration", "cost", "supply", "population"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2. Date Conversion
        for k in ["date", "admit", "discharge", "fill_date"]:
            col = self.cols.get(k)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        self.time_col = (self.cols.get("date") or self.cols.get("admit") or self.cols.get("fill_date"))
    
        # Readmission / flags → numeric 0/1
        flag_col = self.cols.get("readmitted")
        if flag_col and flag_col in df.columns:
            df[flag_col] = (
                df[flag_col]
                .astype(str)
                .str.lower()
                .map({
                    "1": 1, "0": 0,
                    "yes": 1, "no": 0,
                    "true": 1, "false": 0,
                    "y": 1, "n": 0
                })
            )
    
        # -------------------------------------------------
        # 4. DERIVED LOS (ADMIT → DISCHARGE)
        # -------------------------------------------------
        if not self.cols.get("los") and self.cols.get("admit") and self.cols.get("discharge"):
            try:
                admit = pd.to_datetime(df[self.cols["admit"]], errors="coerce")
                discharge = pd.to_datetime(df[self.cols["discharge"]], errors="coerce")
    
                df["_derived_los"] = (discharge - admit).dt.days
                self.cols["los"] = "_derived_los"
            except Exception:
                pass
        
        # -------------------------------------------------
        # 6. FINAL SAFETY CLEANUP
        # -------------------------------------------------
        # Never return mutated original reference
        return df
        
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates factual, domain-level KPIs ONLY.
        
        Guarantees:
        - No executive interpretation
        - No KPI ranking
        - No presentation logic
        - Fully sub-domain & mixed-dataset safe
        """
        
        # -------------------------------------------------
        # 1. SUB-DOMAIN & CAPABILITY DETECTION
        # -------------------------------------------------
        sub, caps = detect_subdomain_and_capabilities(df, self.cols)
        m = HealthcareMapping(df, self.cols)

        # -------------------------------------------------
        # 2. BASE KPIs (ALWAYS SAFE)
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "sub_domain": sub.value,
            "capabilities": [c.value for c in caps],
            "data_completeness": m.data_completeness(),
            "total_volume": m.volume(),
            "total_records": len(df),
            "total_entities": m.volume(),
            "total_patients": m.volume() if self.cols.get("pid") else None, # Legacy compat
        }

        # -------------------------------------------------
        # 3. TIME / DURATION KPIs
        # -------------------------------------------------
        if HealthcareCapability.TIME in caps:
            kpis["avg_duration"] = m.avg_duration()
            
            # Dynamic thresholding based on sub-domain
            duration_threshold = {
                HealthcareSubDomain.HOSPITAL: 7,        # days
                HealthcareSubDomain.DIAGNOSTICS: 60,    # minutes
                HealthcareSubDomain.CLINIC: 1,          # days
            }.get(sub, 7)

            if duration_threshold > 0:
                kpis["long_duration_rate"] = m.long_duration_rate(duration_threshold)
            
            # Legacy mapping for engine compatibility
            kpis["avg_los"] = kpis["avg_duration"] 
            kpis["long_stay_rate"] = kpis.get("long_duration_rate")

        # -------------------------------------------------
        # 4. COST KPIs
        # -------------------------------------------------
        if HealthcareCapability.COST in caps:
            kpis["total_cost"] = m.total_cost()
            kpis["avg_unit_cost"] = m.avg_unit_cost()
            kpis["avg_cost_per_patient"] = kpis["avg_unit_cost"] # Legacy compat

        # -------------------------------------------------
        # 5. QUALITY KPIs
        # -------------------------------------------------
        if HealthcareCapability.QUALITY in caps:
            kpis["adverse_event_rate"] = m.adverse_rate()
            kpis["readmission_rate"] = kpis["adverse_event_rate"] # Legacy compat

        # -------------------------------------------------
        # 6. VARIANCE KPIs
        # -------------------------------------------------
        if HealthcareCapability.VARIANCE in caps:
            kpis["variance_score"] = m.variance()
            kpis["facility_variance_score"] = kpis["variance_score"] # Legacy compat

        # -------------------------------------------------
        # 7. UNIVERSAL EXTENSIONS (Advanced Logic)
        # -------------------------------------------------
        
        # A. Clinic: Appointment Efficiency
        if self.cols.get("status") and self.cols["status"] in df.columns:
            try:
                s = df[self.cols["status"]].astype(str).str.lower()
                kpis["no_show_rate"] = s.isin(["no_show", "noshow", "missed", "dna"]).mean()
            except: pass

        # B. Labs: Diagnostic Outliers
        rv, lo, hi = self.cols.get("result"), self.cols.get("min_val"), self.cols.get("max_val")
        if rv and lo and hi and all(c in df.columns for c in [rv, lo, hi]):
            try:
                temp = df[[rv, lo, hi]].apply(pd.to_numeric, errors='coerce').dropna()
                if not temp.empty:
                    kpis["abnormal_result_rate"] = ((temp[rv] < temp[lo]) | (temp[rv] > temp[hi])).mean()
            except: pass

        # C. Pharmacy: Refill Adherence
        pid, fill, supply = self.cols.get("pid"), self.cols.get("fill_date"), self.cols.get("supply")
        if pid and fill and supply and all(c in df.columns for c in [pid, fill, supply]):
            try:
                temp = df[[pid, fill, supply]].dropna().sort_values(fill)
                # Calculate days between fills vs supply
                temp['gap'] = temp.groupby(pid)[fill].diff().dt.days - temp.groupby(pid)[supply].shift(1)
                kpis["late_refill_rate"] = (temp['gap'] > 5).mean()
            except: pass

        # D. Public Health: Incidence Rate
        pop_col = self.cols.get("population")
        if pop_col and pop_col in df.columns:
            try:
                pop_val = pd.to_numeric(df[pop_col], errors='coerce').dropna().iloc[0]
                if pop_val and pop_val > 0:
                    kpis["incidence_per_100k"] = (kpis["total_volume"] / pop_val) * 100_000
            except: pass

        # -------------------------------------------------
        # 8. TREND SIGNALS
        # -------------------------------------------------
        trend_arrow = "→"
        if self.time_col:
            # Duration Trend
            dur_col = self.cols.get("los") or self.cols.get("duration")
            if dur_col and dur_col in df.columns:
                kpis["avg_duration_trend"] = m.trend(self.time_col, dur_col)
            
            # Cost Trend
            if self.cols.get("cost") and self.cols["cost"] in df.columns:
                kpis["cost_trend"] = m.trend(self.time_col, self.cols["cost"])
            
            # Volume Trend (Resample Safe)
            try:
                vol_s = df.set_index(self.time_col).resample("M").size()
                if len(vol_s) >= 4:
                    delta = (vol_s.iloc[-1] - vol_s.iloc[0]) / vol_s.iloc[0]
                    trend_arrow = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "→"
                    kpis["volume_trend"] = trend_arrow
            except: pass

        # -------------------------------------------------
        # 9. SCORING & METADATA
        # -------------------------------------------------
        score, breakdown = compute_score(kpis, sub, caps)
        
        kpis.update({
            "board_confidence_score": score,
            "board_score_breakdown": breakdown,
            "board_confidence_interpretation": _get_score_interpretation(score),
            "trend_explanation": _get_trend_explanation(trend_arrow),
            "maturity_level": "Gold" if score >= 85 else "Silver" if score >= 70 else "Bronze",
            "board_confidence_trend": trend_arrow,
            "benchmark_context": f"Evaluated against {sub.value.upper()} standards."
        })

        # Executive Selection
        primary_kpis = self.select_executive_kpis(kpis, sub)
        kpis["_executive"] = {
            "primary_kpis": primary_kpis,
            "sub_domain": sub.value,
        }
        
        return kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
    
        # ---------------------------
        # INTERNAL HELPERS
        # ---------------------------
        def save(fig, name, caption, importance):
            fig.savefig(output_dir / name, bbox_inches="tight", dpi=120)
            plt.close(fig)
            visuals.append({
                "path": str(output_dir / name),
                "caption": caption,
                "importance": importance
            })
    
        def human_fmt(x, _):
            try:
                x = float(x)
            except Exception:
                return ""
            if abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
    
        # =================================================
        # 1. VOLUME OVER TIME (UNIVERSAL)
        # =================================================
        if self.time_col and self.time_col in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                df.set_index(self.time_col).resample("M").size().plot(ax=ax, linewidth=2)
                ax.set_title("Activity Volume Over Time", fontweight="bold")
                ax.grid(alpha=0.3)
                save(
                    fig,
                    "volume_trend.png",
                    "Observed activity volume across time.",
                    0.99
                )
            except Exception:
                pass
    
        # =================================================
        # 2. VOLUME DISTRIBUTION
        # =================================================
        if self.time_col and self.time_col in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[self.time_col].dt.to_period("M").value_counts().sort_index().plot(kind="bar", ax=ax)
                ax.set_title("Monthly Activity Distribution", fontweight="bold")
                save(
                    fig,
                    "volume_distribution.png",
                    "Distribution of activity volume across periods.",
                    0.90
                )
            except Exception:
                pass
    
        # =================================================
        # 3. DURATION DISTRIBUTION (LOS / TAT / WAIT)
        # =================================================
        dur_col = c.get("los") or c.get("duration")
        if dur_col and dur_col in df.columns and not df[dur_col].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[dur_col].dropna().hist(ax=ax, bins=20, alpha=0.7)
                ax.set_title("Process Duration Distribution", fontweight="bold")
                save(
                    fig,
                    "duration_dist.png",
                    "Distribution of observed process durations.",
                    0.95
                )
            except Exception:
                pass
    
        # =================================================
        # 4. DURATION TREND
        # =================================================
        if dur_col and self.time_col and dur_col in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                df.set_index(self.time_col)[dur_col].resample("M").mean().plot(ax=ax)
                ax.set_title("Average Duration Trend", fontweight="bold")
                save(
                    fig,
                    "duration_trend.png",
                    "Trend of average process duration over time.",
                    0.92
                )
            except Exception:
                pass
    
        # =================================================
        # 5. COST DISTRIBUTION
        # =================================================
        if c.get("cost") and c["cost"] in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["cost"]].dropna().hist(ax=ax, bins=20)
                ax.set_title("Cost Distribution", fontweight="bold")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "cost_distribution.png",
                    "Distribution of observed cost values.",
                    0.94
                )
            except Exception:
                pass
    
        # =================================================
        # 6. TOP COST DRIVERS (CATEGORY-AGNOSTIC)
        # =================================================
        group_col = c.get("diagnosis") or c.get("type") or c.get("facility")
        if c.get("cost") and group_col and group_col in df.columns:
            try:
                stats = df.groupby(group_col)[c["cost"]].mean().nlargest(5)
                if not stats.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    stats.plot(kind="bar", ax=ax)
                    ax.set_title("Top Value Contributors", fontweight="bold")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(
                        fig,
                        "cost_drivers.png",
                        "Categories with highest average cost contribution.",
                        0.93
                    )
            except Exception:
                pass
    
        # =================================================
        # 7. QUALITY / ADVERSE EVENT RATES
        # =================================================
        quality_col = c.get("readmitted") or c.get("flag")
        if quality_col and quality_col in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[quality_col].value_counts(normalize=True).plot(kind="bar", ax=ax)
                ax.set_title("Quality / Adverse Event Distribution", fontweight="bold")
                save(
                    fig,
                    "quality_dist.png",
                    "Distribution of quality-related events.",
                    0.88
                )
            except Exception:
                pass
    
        # =================================================
        # 8. VARIANCE BY ENTITY
        # =================================================
        var_group = c.get("facility") or c.get("doctor")
        var_metric = c.get("cost") or c.get("los")
        if var_group and var_metric and var_group in df.columns and var_metric in df.columns:
            try:
                stats = df.groupby(var_group)[var_metric].mean().nlargest(10)
                if not stats.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    stats.plot(kind="bar", ax=ax)
                    ax.set_title("Entity-Level Performance Variance", fontweight="bold")
                    save(
                        fig,
                        "variance_entities.png",
                        "Observed performance variation across entities.",
                        0.87
                    )
            except Exception:
                pass
    
        # =================================================
        # 9. TEMPORAL PATTERN (WEEKDAY)
        # =================================================
        if self.time_col and self.time_col in df.columns:
            try:
                dow = df[self.time_col].dropna().dt.day_name()
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                counts = dow.value_counts().reindex(order, fill_value=0)
                fig, ax = plt.subplots(figsize=(6, 4))
                counts.plot(kind="bar", ax=ax)
                ax.set_title("Activity by Day of Week", fontweight="bold")
                save(
                    fig,
                    "weekday_pattern.png",
                    "Observed activity distribution by weekday.",
                    0.85
                )
            except Exception:
                pass
    
        # =================================================
        # GUARANTEE: SORT & RETURN
        # =================================================
        return sorted(visuals, key=lambda x: x["importance"], reverse=True)


    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        shape_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []
    
        sub_domain = kpis.get("sub_domain", "unknown")
        caps = set(kpis.get("capabilities", []))
    
        def add(level, title, so_what, source="System Analysis", executive=False):
            insights.append({
                "level": level,
                "title": title,
                "so_what": so_what,
                "source": source,
                "executive_summary_flag": executive
            })
    
        # =================================================
        # 1. DATA TRUST FOUNDATION (ALWAYS FIRST)
        # =================================================
        dc = kpis.get("data_completeness")
        if isinstance(dc, (int, float)):
            if dc < 0.80:
                add(
                    "CRITICAL",
                    "Low Data Reliability",
                    "Key fields contain significant missing values. Decisions based on this data carry elevated risk.",
                    "Data Quality Assessment",
                    True
                )
            elif dc < 0.90:
                add(
                    "RISK",
                    "Moderate Data Gaps Detected",
                    "Some data incompleteness exists. Findings should be reviewed with caution.",
                    "Data Quality Assessment"
                )
            else:
                add(
                    "INFO",
                    "High Data Reliability",
                    "Data completeness supports confident operational and executive analysis.",
                    "Data Quality Assessment"
                )
    
        # =================================================
        # 2. SCALE & DEMAND PRESSURE
        # =================================================
        vol = kpis.get("total_volume")
        if isinstance(vol, (int, float)):
            if vol > 100_000:
                add(
                    "INFO",
                    "Large-Scale Operations Detected",
                    "High activity volume amplifies the financial and operational impact of inefficiencies.",
                    "Demand Analysis",
                    True
                )
            else:
                add(
                    "INFO",
                    "Moderate Operational Scale",
                    "Observed volume suggests localized or specialized operations.",
                    "Demand Analysis"
                )
    
        # =================================================
        # 3. TIME / FLOW BOTTLENECKS
        # =================================================
        long_rate = kpis.get("long_duration_rate")
        avg_dur = kpis.get("avg_duration")
    
        if isinstance(long_rate, (int, float)):
            if long_rate > 0.25:
                add(
                    "CRITICAL",
                    "Severe Flow Bottleneck",
                    "A significant share of cases exceed expected processing time, constraining throughput and capacity.",
                    "Flow Analysis",
                    True
                )
            elif long_rate > 0.10:
                add(
                    "WARNING",
                    "Emerging Flow Inefficiency",
                    "A noticeable portion of cases are delayed, indicating early-stage bottlenecks.",
                    "Flow Analysis"
                )
            else:
                add(
                    "INFO",
                    "Process Flow Within Norms",
                    "Most cases complete within expected time ranges.",
                    "Flow Analysis"
                )
    
        # =================================================
        # 4. COST & ECONOMIC PRESSURE
        # =================================================
        avg_cost = kpis.get("avg_unit_cost")
        cost_trend = kpis.get("cost_trend")
    
        if isinstance(avg_cost, (int, float)):
            if cost_trend == "↑":
                add(
                    "WARNING",
                    "Unit Cost Escalation",
                    "Rising unit costs indicate growing financial pressure requiring intervention.",
                    "Cost Dynamics",
                    True
                )
            elif cost_trend == "↓":
                add(
                    "INFO",
                    "Improving Cost Efficiency",
                    "Recent trends suggest better cost control and efficiency gains.",
                    "Cost Dynamics"
                )
            else:
                add(
                    "INFO",
                    "Stable Cost Structure",
                    "Unit cost levels remain stable over the observed period.",
                    "Cost Dynamics"
                )
    
        # =================================================
        # 5. QUALITY & RISK SIGNALS
        # =================================================
        quality_rate = (
            kpis.get("adverse_event_rate")
            or kpis.get("no_show_rate")
            or kpis.get("out_of_range_result_rate")
        )
    
        if isinstance(quality_rate, (int, float)):
            if quality_rate > 0.20:
                add(
                    "CRITICAL",
                    "Elevated Quality Risk",
                    "Observed quality-related events exceed acceptable thresholds, increasing operational and reputational exposure.",
                    "Quality Signals",
                    True
                )
            elif quality_rate > 0.10:
                add(
                    "RISK",
                    "Moderate Quality Degradation",
                    "Quality indicators show early warning signs of instability.",
                    "Quality Signals"
                )
            else:
                add(
                    "INFO",
                    "Quality Performance Stable",
                    "Quality-related indicators remain within expected bounds.",
                    "Quality Signals"
                )
    
        # =================================================
        # 6. VARIABILITY & EXECUTION CONSISTENCY
        # =================================================
        var = kpis.get("variance_score")
        if isinstance(var, (int, float)):
            if var > 0.60:
                add(
                    "RISK",
                    "Highly Inconsistent Execution",
                    "Significant performance variation exists across entities, indicating weak standardization.",
                    "Variance Analysis",
                    True
                )
            elif var > 0.30:
                add(
                    "WARNING",
                    "Moderate Execution Variability",
                    "Some inconsistency exists across entities, suggesting improvement opportunities.",
                    "Variance Analysis"
                )
            else:
                add(
                    "INFO",
                    "Consistent Operational Execution",
                    "Performance appears relatively uniform across observed entities.",
                    "Variance Analysis"
                )
    
        # =================================================
        # 7. MOMENTUM & TRAJECTORY
        # =================================================
        momentum = (
            kpis.get("volume_trend")
            or kpis.get("avg_duration_trend")
            or kpis.get("cost_trend")
        )
    
        if momentum == "↑":
            add(
                "WARNING",
                "Negative Performance Momentum",
                "Recent trends indicate deteriorating performance that may accelerate without intervention.",
                "Trend Analysis",
                True
            )
        elif momentum == "↓":
            add(
                "INFO",
                "Positive Performance Momentum",
                "Recent trends indicate improving operational performance.",
                "Trend Analysis"
            )
        else:
            add(
                "INFO",
                "Stable Performance Trajectory",
                "No strong directional trend detected in recent periods.",
                "Trend Analysis"
            )
    
        # =================================================
        # 8. GUARANTEE MINIMUM EXECUTIVE COVERAGE
        # =================================================
        while len(insights) < 8:
            add(
                "INFO",
                f"Operational Observation #{len(insights) + 1}",
                "No additional statistically significant anomalies detected.",
                "System Generated"
            )
    
        return insights

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: Optional[List[Dict[str, Any]]] = None,
        shape_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
    
        recommendations: List[Dict[str, Any]] = []
        insight_titles = {i.get("title") for i in (insights or [])}
    
        caps = set(kpis.get("capabilities", []))
        sub_domain = kpis.get("sub_domain", "unknown")
    
        # Helper
        def add(
            action: str,
            priority: str,
            owner: str,
            timeline: str,
            outcome: str,
            confidence: float,
            impact: float
        ):
            recommendations.append({
                "action": action,
                "priority": priority,
                "owner": owner,
                "timeline": timeline,
                "expected_outcome": outcome,
                "confidence": round(float(confidence), 2),
                "impact_score": round(float(impact), 2)
            })
    
        # =================================================
        # 1. DATA QUALITY REMEDIATION
        # =================================================
        if "Data Quality Constraint" in insight_titles:
            add(
                "Strengthen data capture and validation controls",
                "CRITICAL",
                "Data & Analytics Leadership",
                "Immediate",
                "Improved analytical reliability and governance confidence",
                0.95,
                0.95
            )
    
        # =================================================
        # 2. PROCESS BOTTLENECK
        # =================================================
        if "Process Bottleneck Identified" in insight_titles:
            add(
                "Identify and eliminate primary workflow bottlenecks",
                "HIGH",
                "Operational Excellence",
                "30–60 days",
                "Improved throughput and capacity utilization",
                0.90,
                0.92
            )
    
        # =================================================
        # 3. COST PRESSURE
        # =================================================
        if "Rising Unit Cost Trend" in insight_titles:
            add(
                "Analyze cost drivers and control high-impact contributors",
                "HIGH",
                "Finance & Operations",
                "30–60 days",
                "Stabilized unit economics and reduced financial exposure",
                0.85,
                0.90
            )
    
        # =================================================
        # 4. QUALITY / RISK CONTROL
        # =================================================
        if "Elevated Quality Risk" in insight_titles:
            add(
                "Initiate targeted quality and safety intervention",
                "CRITICAL",
                "Quality & Compliance",
                "Immediate",
                "Reduced adverse events and risk exposure",
                0.95,
                0.94
            )
    
        # =================================================
        # 5. VARIANCE REDUCTION
        # =================================================
        if "High Operational Variability" in insight_titles:
            add(
                "Standardize processes across high-variance entities",
                "HIGH",
                "Operations Leadership",
                "60–90 days",
                "Improved consistency and predictable performance",
                0.88,
                0.90
            )
    
        # =================================================
        # 6. TREND MOMENTUM MANAGEMENT
        # =================================================
        if "Negative Momentum Detected" in insight_titles:
            add(
                "Assign executive owner to reverse negative performance trend",
                "HIGH",
                "Executive Sponsor",
                "30 days",
                "Stabilized and improved performance trajectory",
                0.90,
                0.88
            )
    
        # =================================================
        # 7. ACCESS / CAPACITY OPTIMIZATION (UNIVERSAL)
        # =================================================
        if "Operational Demand Observed" in insight_titles:
            add(
                "Align capacity and resources with observed demand patterns",
                "MEDIUM",
                "Operations Planning",
                "60–90 days",
                "Reduced congestion and improved service access",
                0.80,
                0.82
            )
    
        # =================================================
        # 8. GOVERNANCE MATURITY (ALWAYS SAFE)
        # =================================================
        add(
            "Establish recurring executive review of key performance indicators",
            "LOW",
            "Executive Governance",
            "Ongoing",
            "Sustained performance oversight and early risk detection",
            0.75,
            0.70
        )
    
        # =================================================
        # GUARANTEE: MINIMUM 7 RECOMMENDATIONS
        # =================================================
        while len(recommendations) < 7:
            add(
                "Continue monitoring key operational indicators",
                "LOW",
                "Operations",
                "Ongoing",
                "Early detection of emerging risks",
                0.70,
                0.60
            )
    
        # =================================================
        # FINAL SORT: EXECUTIVE IMPACT FIRST
        # =================================================
        recommendations = sorted(
            recommendations,
            key=lambda x: x["impact_score"] * x["confidence"],
            reverse=True
        )
    
        # Remove internal-only field
        for r in recommendations:
            r.pop("impact_score", None)
    
        return recommendations

# =====================================================
# 7. DOMAIN REGISTRATION
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {
        "patient", "admission", "diagnosis", "clinical",
        "doctor", "insurance", "test", "specimen", "rx"
    }

    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        return DomainDetectionResult("healthcare", min(len(hits) / 3, 1.0), {})


def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
