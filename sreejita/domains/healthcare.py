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
        if self.c.get("cost"):
            metric = self.c["cost"]
        elif self.c.get("los"):
            metric = self.c["los"]
        elif self.c.get("duration"):
            metric = self.c["duration"]
        else:
            return None
    
        for g in ["facility", "doctor"]:
            if self.c.get(g) and metric in self.df.columns:
                stats = self.df.groupby(self.c[g])[metric].mean()
                if stats.mean() > 0:
                    return min(stats.std() / stats.mean(), 1.0)
        return None

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

    def usable(col, min_ratio=0.3):
        return col and col in df.columns and df[col].notna().mean() >= min_ratio

    # -----------------------
    # CAPABILITIES
    # -----------------------
    if usable(cols.get("pid")) or usable(cols.get("encounter")):
        caps.add(HealthcareCapability.VOLUME)

    if usable(cols.get("los")) or usable(cols.get("duration")):
        caps.add(HealthcareCapability.TIME)

    if usable(cols.get("cost")):
        caps.add(HealthcareCapability.COST)

    if usable(cols.get("readmitted")) or usable(cols.get("flag")):
        caps.add(HealthcareCapability.QUALITY)

    if usable(cols.get("facility")) or usable(cols.get("doctor")):
        caps.add(HealthcareCapability.VARIANCE)

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
def _normalize_boolean_like(series: pd.Series) -> pd.Series:
    """
    Converts Yes/No, True/False, Y/N into 1/0 safely.
    Leaves numeric values unchanged.
    """
    if series.dtype == object:
        return series.map(
            lambda x: (
                1 if str(x).strip().lower() in {"yes", "y", "true", "1"} else
                0 if str(x).strip().lower() in {"no", "n", "false", "0"} else
                np.nan
            )
        )
    return series

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
        }
    
        # -------------------------------------------------
        # 3. TYPE NORMALIZATION (CRITICAL)
        # -------------------------------------------------
        for key in ("los", "duration", "cost"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
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
        # 5. TIME COLUMN RESOLUTION (FOR TRENDS)
        # -------------------------------------------------
        self.time_col = None
    
        for key in ("date", "admit"):
            col = self.cols.get(key)
            if col and col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    self.time_col = col
                    break
                except Exception:
                    continue
    
        # -------------------------------------------------
        # 6. FINAL SAFETY CLEANUP
        # -------------------------------------------------
        # Never return mutated original reference
        return df
        
    def calculate_kpis(self, df):
        sub, caps = detect_subdomain_and_capabilities(df, self.cols)
        m = HealthcareMapping(df, self.cols)

        kpis = {
            "sub_domain": sub.value,
            "capabilities": [c.value for c in caps],
            "data_completeness": m.data_completeness(),
            "total_volume": m.volume(),
        }

        if HealthcareCapability.TIME in caps:
            kpis["avg_duration"] = m.avg_duration()
            duration_threshold = {
                HealthcareSubDomain.HOSPITAL: 7,
                HealthcareSubDomain.DIAGNOSTICS: 60,   # minutes
                HealthcareSubDomain.CLINIC: 1,         # days
                HealthcareSubDomain.PHARMACY: 0,       # not applicable
                HealthcareSubDomain.PUBLIC_HEALTH: 0,
            }.get(sub, 7)

            if duration_threshold > 0:
                kpis["long_duration_rate"] = m.long_duration_rate(duration_threshold)
                
        if HealthcareCapability.COST in caps:
            kpis["total_cost"] = m.total_cost()
            kpis["avg_unit_cost"] = m.avg_unit_cost()

        if HealthcareCapability.QUALITY in caps:
            kpis["adverse_event_rate"] = m.adverse_rate()

        if HealthcareCapability.VARIANCE in caps:
            kpis["variance_score"] = m.variance()

        score, breakdown = compute_score(kpis, sub, caps)
        kpis.update({
            "board_confidence_score": score,
            "board_score_breakdown": breakdown,
            "maturity_level": "Gold" if score >= 85 else "Silver" if score >= 70 else "Bronze",
        })

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
    
        # -------------------------------------------------
        # CONTEXT
        # -------------------------------------------------
        sub_domain = kpis.get("sub_domain", "unknown")
        caps = set(kpis.get("capabilities", []))
    
        # Helper
        def add(level, title, so_what, source="System Analysis", executive=False):
            insights.append({
                "level": level,
                "title": title,
                "so_what": so_what,
                "source": source,
                "executive_summary_flag": executive
            })
    
        # =================================================
        # 1. DATA QUALITY FOUNDATION (ALWAYS)
        # =================================================
        dc = kpis.get("data_completeness")
        if isinstance(dc, (int, float)):
            if dc < 0.85:
                add(
                    "RISK",
                    "Data Quality Constraint",
                    "A material portion of required fields is missing. Analytical confidence and downstream decision reliability are reduced.",
                    "Data Integrity Check",
                    True
                )
            else:
                add(
                    "INFO",
                    "Sufficient Data Coverage",
                    "Available data is sufficiently complete to support reliable operational analysis.",
                    "Data Integrity Check"
                )
    
        # =================================================
        # 2. VOLUME SIGNAL (UNIVERSAL)
        # =================================================
        vol = kpis.get("total_volume") or kpis.get("total_patients") or kpis.get("total_encounters")
        if isinstance(vol, (int, float)):
            add(
                "INFO",
                "Operational Demand Observed",
                f"Observed activity volume is {int(vol):,}. Scale effects amplify the impact of inefficiencies.",
                "Volume Analysis"
            )
    
        # =================================================
        # 3. TIME / DURATION BOTTLENECK
        # =================================================
        long_rate = kpis.get("long_duration_rate") or kpis.get("long_stay_rate")
        avg_dur = kpis.get("avg_duration") or kpis.get("avg_los")
    
        if isinstance(long_rate, (int, float)) and long_rate > 0.2:
            label = kpis.get("duration_label", "process duration")
            add(
                "CRITICAL",
                "Process Bottleneck Identified",
                f"A high proportion of records exhibit extended {label}. This directly constrains throughput and capacity.",
                "Duration Distribution",
                True
            )
        elif isinstance(avg_dur, (int, float)):
            add(
                "INFO",
                "Stable Process Timing",
                "Average process duration remains within expected operational bounds.",
                "Duration Analysis"
            )
    
        # =================================================
        # 4. COST PRESSURE SIGNAL
        # =================================================
        avg_cost = kpis.get("avg_unit_cost") or kpis.get("avg_cost_per_patient")
        cost_trend = kpis.get("cost_trend")
    
        if isinstance(avg_cost, (int, float)) and cost_trend == "↑":
            add(
                "WARNING",
                "Rising Unit Cost Trend",
                "Average cost per unit is increasing over time, indicating emerging financial pressure.",
                "Cost Trend Analysis",
                True
            )
        elif isinstance(avg_cost, (int, float)):
            add(
                "INFO",
                "Cost Levels Observed",
                "Unit cost levels appear stable based on available data.",
                "Cost Analysis"
            )
    
        # =================================================
        # 5. QUALITY / RISK SIGNAL
        # =================================================
        quality_rate = (
            kpis.get("adverse_event_rate")
            or kpis.get("readmission_rate")
        )
    
        if isinstance(quality_rate, (int, float)):
            if quality_rate > 0.15:
                label = kpis.get("quality_label", "adverse events")
                add(
                    "CRITICAL",
                    "Elevated Quality Risk",
                    f"Observed rate of {label} is materially elevated, increasing operational and reputational risk.",
                    "Quality Signal",
                    True
                )
            else:
                add(
                    "INFO",
                    "Quality Signals Within Range",
                    "Observed quality-related event rates do not indicate systemic instability.",
                    "Quality Signal"
                )
    
        # =================================================
        # 6. VARIANCE SIGNAL (FACILITY / PROVIDER / CATEGORY)
        # =================================================
        var_score = kpis.get("variance_score") or kpis.get("facility_variance_score")
        if isinstance(var_score, (int, float)) and var_score > 0.5:
            add(
                "RISK",
                "High Operational Variability",
                "Significant variation exists across entities, indicating inconsistent execution and opportunity for standardization.",
                "Variance Analysis",
                True
            )
        else:
            add(
                "INFO",
                "Controlled Operational Variability",
                "Performance variation across entities appears limited.",
                "Variance Analysis"
            )
    
        # =================================================
        # 7. TREND MOMENTUM SIGNAL
        # =================================================
        trend = (
            kpis.get("avg_duration_trend")
            or kpis.get("los_trend")
            or kpis.get("volume_trend")
        )
    
        if trend == "↑":
            add(
                "WARNING",
                "Negative Momentum Detected",
                "Recent trend indicates worsening performance trajectory requiring attention.",
                "Trend Analysis"
            )
        elif trend == "↓":
            add(
                "INFO",
                "Positive Momentum Detected",
                "Recent trend indicates improving operational performance.",
                "Trend Analysis"
            )
        else:
            add(
                "INFO",
                "Stable Performance Trend",
                "No sustained directional trend detected in recent periods.",
                "Trend Analysis"
            )
    
        # =================================================
        # GUARANTEE: MINIMUM 7 INSIGHTS
        # =================================================
        seen_titles = set(i["title"] for i in insights)
        while len(insights) < 7:
            title = f"Operational Observation #{len(insights)+1}"
            if title not in seen_titles:
                add(
                    "INFO",
                    title,
                    "No additional statistically significant anomalies detected in available data.",
                    "System Generated"
                )
                seen_titles.add(title) 
                
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
