import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from matplotlib.ticker import FuncFormatter
from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# HEALTHCARE DOMAIN MAPPING (DOMAIN-SIDE ONLY)
# =====================================================
class HealthcareMapping:
    """
    Healthcare FACT extraction layer.

    RULES (STRICT):
    - NO benchmarks
    - NO scoring
    - NO thresholds
    - NO executive interpretation
    - NO rendering logic

    This class only answers:
    "What does the data say?"
    """

    def __init__(self, df: pd.DataFrame, cols: Dict[str, str]):
        self.df = df
        self.c = cols

    # =================================================
    # DATA QUALITY
    # =================================================
    def data_completeness(self) -> float:
        critical = [
            self.c.get("pid"),
            self.c.get("los"),
            self.c.get("cost"),
            self.c.get("diagnosis"),
            self.c.get("facility"),
        ]
        critical = [c for c in critical if c and c in self.df.columns]

        if not critical:
            return 0.0

        return round(1 - self.df[critical].isna().mean().mean(), 2)

    # =================================================
    # VOLUME SIGNALS
    # =================================================
    def total_patients(self) -> int:
        if self.c.get("pid") and self.c["pid"] in self.df.columns:
            return self.df[self.c["pid"]].nunique()
        return len(self.df)

    def total_encounters(self) -> int:
        if self.c.get("encounter") and self.c["encounter"] in self.df.columns:
            return self.df[self.c["encounter"]].nunique()
        return len(self.df)

    def visits_per_patient(self) -> Optional[float]:
        if self.c.get("pid") and self.c.get("encounter"):
            patients = self.total_patients()
            encounters = self.total_encounters()
            if patients > 0:
                return encounters / patients
        return None

    # =================================================
    # LENGTH OF STAY SIGNALS
    # =================================================
    def avg_los(self) -> Optional[float]:
        if self.c.get("los") and self.c["los"] in self.df.columns:
            return self.df[self.c["los"]].mean()
        return None

    def long_stay_rate(self, threshold: float) -> Optional[float]:
        if self.c.get("los") and self.c["los"] in self.df.columns:
            return (self.df[self.c["los"]] > threshold).mean()
        return None

    # =================================================
    # QUALITY SIGNALS
    # =================================================
    def readmission_rate(self) -> Optional[float]:
        if (
            self.c.get("readmitted")
            and self.c["readmitted"] in self.df.columns
            and pd.api.types.is_numeric_dtype(self.df[self.c["readmitted"]])
        ):
            return self.df[self.c["readmitted"]].mean()
        return None

    # =================================================
    # COST SIGNALS
    # =================================================
    def total_cost(self) -> Optional[float]:
        if self.c.get("cost") and self.c["cost"] in self.df.columns:
            return self.df[self.c["cost"]].sum()
        return None

    def avg_cost_per_patient(self) -> Optional[float]:
        if self.c.get("cost") and self.c["cost"] in self.df.columns:
            return self.df[self.c["cost"]].mean()
        return None

    def avg_cost_per_day(self) -> Optional[float]:
        if (
            self.c.get("cost")
            and self.c.get("los")
            and self.c["cost"] in self.df.columns
            and self.c["los"] in self.df.columns
        ):
            los_mean = self.df[self.c["los"]].mean()
            if los_mean and los_mean > 0:
                return self.df[self.c["cost"]].mean() / los_mean
        return None

    # =================================================
    # VARIANCE SIGNALS
    # =================================================
    def facility_variance(self) -> Optional[float]:
        if self.c.get("facility") and self.c.get("cost"):
            fac = self.c["facility"]
            cost = self.c["cost"]
            if fac in self.df.columns and cost in self.df.columns:
                stats = self.df.groupby(fac)[cost].mean()
                if stats.mean() > 0:
                    return min(stats.std() / stats.mean(), 1.0)
        return None

    def provider_variance(self) -> Optional[float]:
        if self.c.get("doctor") and self.c.get("los"):
            doc = self.c["doctor"]
            los = self.c["los"]
            if doc in self.df.columns and los in self.df.columns:
                stats = self.df.groupby(doc)[los].mean()
                if stats.mean() > 0:
                    return min(stats.std() / stats.mean(), 1.0)
        return None

    # =================================================
    # TEMPORAL SIGNALS
    # =================================================
    def trend(self, time_col: str, metric_key: str) -> str:
        """
        Returns symbolic trend only: ↑ ↓ →
        """
        if not time_col or metric_key not in self.c:
            return "→"

        metric = self.c.get(metric_key)
        if metric not in self.df.columns or time_col not in self.df.columns:
            return "→"

        df = self.df.dropna(subset=[metric]).sort_values(time_col)
        if len(df) < 10:
            return "→"

        cut = int(len(df) * 0.8)
        hist = df.iloc[:cut][metric].mean()
        recent = df.iloc[cut:][metric].mean()

        if not hist or hist == 0:
            return "→"

        delta = (recent - hist) / hist
        if delta > 0.05:
            return "↑"
        if delta < -0.05:
            return "↓"
        return "→"

    def weekend_admission_rate(self, time_col: str) -> Optional[float]:
        if time_col and time_col in self.df.columns:
            dow = self.df[time_col].dt.dayofweek
            return dow.isin([5, 6]).mean()
        return None

    def internal_cost_benchmark(self, multiplier: float = 1.1) -> Optional[float]:
        if self.c.get("cost") and self.c["cost"] in self.df.columns:
            return self.df[self.c["cost"]].median() * multiplier
        return None

# =====================================================
# ENGINE & GOVERNANCE IMPORTS (SEPARATED)
# =====================================================
from sreejita.narrative.benchmarks import HEALTHCARE_THRESHOLDS, HEALTHCARE_EXTERNAL_LIMITS

VISUAL_BENCHMARK_LOS = HEALTHCARE_THRESHOLDS.get("benchmark_los", 5.0)

# =====================================================
# CONSTANTS & STANDARDS
# =====================================================
class CareContext(str, Enum):
    INPATIENT = "inpatient"     # Hospitals (Has LOS)
    OUTPATIENT = "outpatient"   # Clinics (Visits, No Bed)
    EMERGENCY = "emergency"     # ED (Triage, Acuity, Arrival)
    DIAGNOSTIC = "diagnostic"   # Labs/Tests (Encounter, No Stay)
    AGGREGATED = "aggregated"   # Monthly Summaries
    UNKNOWN = "unknown"

class DatasetShape(str, Enum):
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    UNKNOWN = "unknown"

# 2️⃣ CONTEXT-AWARE KPI EXPECTATIONS
EXPECTED_METRICS = {
    CareContext.INPATIENT: {"avg_los", "readmission_rate", "long_stay_rate"},
    CareContext.OUTPATIENT: {"total_patients", "avg_cost_per_patient"},
    CareContext.EMERGENCY: {"total_patients", "avg_los"},
    CareContext.DIAGNOSTIC: {"total_encounters", "data_completeness"},
    CareContext.AGGREGATED: {}, 
    CareContext.UNKNOWN: {}
}

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    score = {k: 0 for k in DatasetShape}

    if any(x in c for c in cols for x in ["patient", "mrn", "pid", "id", "name", "encounter", "visit"]): 
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
    if any(x in c for c in cols for x in ["admit", "admission", "date", "joining", "test_date", "arrival"]): 
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
    
    if any(x in c for c in cols for x in ["total", "volume", "census", "visits"]): 
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
    
    if any(x in c for c in cols for x in ["revenue", "bill", "cost", "fee", "amount"]): 
        score[DatasetShape.FINANCIAL_SUMMARY] += 2

    best = max(score, key=score.get)
    if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 3: best = DatasetShape.ROW_LEVEL_CLINICAL
    if score[best] == 0: best = DatasetShape.UNKNOWN
    return {"shape": best, "score": score}

def _get_trend_explanation(trend_arrow):
    """Explains WHY the trend is what it is."""
    if trend_arrow == "→": return "Flat trend indicates no sustained improvement over prior period."
    if trend_arrow == "↑": return "Increasing trend requires immediate variance control."
    if trend_arrow == "↓": return "Decreasing trend indicates operational improvements taking effect."
    return "Trend data insufficient."

def _get_score_interpretation(score):
    """Explains the Governance Implication of the score."""
    if score < 50: 
        return "GOVERNANCE RISK: Performance is reactive and structurally unstable. Requires immediate executive intervention."
    if score < 70: 
        return "UNSTABLE: High variance detected. Governance priority: Standardization and Variance Control."
    if score < 85: 
        return "CONTROLLED: Processes stable but reactive. Governance priority: Optimization."
    return "PREDICTIVE: Best-in-class operations."

# =====================================================
# 3️⃣ BOARD CONFIDENCE (Weighted & Explained)
# =====================================================
def _compute_board_confidence_score(kpis: Dict[str, Any], context: CareContext) -> Tuple[int, Dict[str, int]]:
    score = 100
    breakdown = {} 
    t = HEALTHCARE_THRESHOLDS
    
    # --- AGGREGATED PATH ---
    if context == CareContext.AGGREGATED:
        score = 70
        breakdown["Aggregated Data Limit"] = -30
        
        # Bonuses
        if kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] <= kpis["benchmark_los"]: 
            score += 10
            breakdown["Bonus: LOS within Target"] = 10
        
        if kpis.get("readmission_rate") and kpis["readmission_rate"] <= 0.1:
            score += 10
            breakdown["Bonus: Optimal Readmissions"] = 10
            
        final_score = min(score, 85)
        if final_score < score:
            diff = final_score - score
            breakdown["Cap: Aggregated Ceiling"] = diff
            
        return final_score, breakdown

    # --- ROW LEVEL PATH ---
    
    # A. Check EXPECTED metrics
    required = EXPECTED_METRICS.get(context, set())
    for metric in required:
        if kpis.get(metric) is None:
            score -= 10
            breakdown[f"Missing Metric: {metric}"] = -10

    # B. Facility Variance
    if kpis.get("facility_variance_score", 0) > 0.5:
        score -= 10
        breakdown["High Facility Variance"] = -10

    # C. Threshold Penalties
    if kpis.get("long_stay_rate", 0) >= t.get("long_stay_rate_critical", 0.3): 
        score -= 25
        breakdown["Critical Long Stay Rate"] = -25
    elif kpis.get("long_stay_rate", 0) >= t.get("long_stay_rate_warning", 0.2): 
        score -= 15
        breakdown["High Long Stay Rate"] = -15

    if kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"]: 
        score -= 10
        breakdown["LOS Exceeds Benchmark"] = -10

    if kpis.get("readmission_rate", 0) >= t.get("readmission_critical", 0.15): 
        score -= 20
        breakdown["Critical Readmission Rate"] = -20
    
    if kpis.get("avg_cost_per_patient") and kpis.get("benchmark_cost"):
        if kpis["avg_cost_per_patient"] > kpis["benchmark_cost"]: 
            score -= 10
            breakdown["Cost > Benchmark"] = -10

    # D. Diagnostic Specific
    if context == CareContext.DIAGNOSTIC:
        if kpis.get("data_completeness", 0) < 0.9: 
            score -= 20
            breakdown["Low Data Completeness"] = -20

    if not breakdown and score == 100:
        breakdown["Perfect Operational State"] = 0

    return max(score, 0), breakdown

def _healthcare_maturity_level(board_score: int) -> str:
    return "Gold" if board_score >= 85 else "Silver" if board_score >= 70 else "Bronze"

# =====================================================
# HEALTHCARE DOMAIN (FACT ENGINE)
# =====================================================
class HealthcareDomain(BaseDomain):
    name = "healthcare"
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.shape_info = detect_dataset_shape(df)
        self.shape = self.shape_info["shape"]
        
        # UNIVERSAL COLUMN MAPPING
        self.cols = {
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "name") or resolve_column(df, "mrn"),
            "encounter": resolve_column(df, "encounter_id") or resolve_column(df, "visit_id") or resolve_column(df, "test_id"),
            "facility": resolve_column(df, "hospital_id") or resolve_column(df, "facility_code") or resolve_column(df, "location") or resolve_column(df, "center"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition") or resolve_column(df, "condition"),
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost") or resolve_column(df, "charges"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "readmission"),
            "volume": resolve_column(df, "total_patients") or resolve_column(df, "visits"),
            "avg_los": resolve_column(df, "avg_los"),
            "admit": resolve_column(df, "date_of_admission") or resolve_column(df, "admission_date") or resolve_column(df, "joining_date") or resolve_column(df, "test_date") or resolve_column(df, "arrival_date"),
            "discharge": resolve_column(df, "discharge_date") or resolve_column(df, "date_of_discharge"),
            "payer": resolve_column(df, "insurance_provider") or resolve_column(df, "payer") or resolve_column(df, "insurance"),
            "age": resolve_column(df, "age") or resolve_column(df, "dob"),
            "type": resolve_column(df, "admission_type") or resolve_column(df, "type"),
        }

        # 1️⃣ EXPLICIT CARE-CONTEXT RESOLUTION
        cols_lower = [c.lower() for c in df.columns]
        
        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL:
            is_ed = any(x in c for c in cols_lower for x in ["triage", "acuity", "arrival", "ed_visit", "er_admit"])
            
            if is_ed:
                self.care_context = CareContext.EMERGENCY
            elif self.cols["los"] or (self.cols["admit"] and self.cols["discharge"]):
                self.care_context = CareContext.INPATIENT
            elif self.cols["encounter"] and not self.cols["los"]:
                self.care_context = CareContext.DIAGNOSTIC
            else:
                self.care_context = CareContext.OUTPATIENT
                
        elif self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            self.care_context = CareContext.AGGREGATED
        else:
            self.care_context = CareContext.UNKNOWN
        
        # Numeric Conversion
        for k in ["los", "cost", "volume", "avg_los", "age"]:
            if self.cols.get(k) in df.columns: 
                df[self.cols[k]] = pd.to_numeric(df[self.cols[k]], errors='coerce')

        # Derived LOS
        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL and not self.cols["los"] and self.cols["admit"] and self.cols["discharge"]:
            try:
                a = pd.to_datetime(df[self.cols["admit"]], errors="coerce")
                d = pd.to_datetime(df[self.cols["discharge"]], errors="coerce")
                df["derived_los"] = (d - a).dt.days
                self.cols["los"] = "derived_los"
            except: pass

        if self.cols["admit"]:
            df[self.cols["admit"]] = pd.to_datetime(df[self.cols["admit"]], errors="coerce")
            self.time_col = self.cols["admit"]
            df = df.sort_values(self.time_col)
        else: self.time_col = None

        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        UNIVERSAL KPI ORCHESTRATOR (Domain-safe, Engine-governed)
        """
        
        # -------------------------------------------------
        # 0. Normalize context
        # -------------------------------------------------
        raw_kpis = {
            "dataset_shape": self.shape.value if hasattr(self.shape, "value") else self.shape,
            "care_context": self.care_context.value if hasattr(self.care_context, "value") else self.care_context,
        }
        
        c = self.cols
        mapping = HealthcareMapping(df, c)
        
        # -------------------------------------------------
        # 1. Data integrity (FACT)
        # -------------------------------------------------
        raw_kpis["data_completeness"] = mapping.data_completeness()
        
        # -------------------------------------------------
        # 2. Volume & population (FACT)
        # -------------------------------------------------
        raw_kpis["total_patients"] = mapping.total_patients()
        raw_kpis["total_encounters"] = mapping.total_encounters()
        raw_kpis["visits_per_patient"] = mapping.visits_per_patient()
        
        # -------------------------------------------------
        # 3. Length of Stay (FACT)
        # -------------------------------------------------
        raw_kpis["avg_los"] = mapping.avg_los()
        los_threshold = HEALTHCARE_THRESHOLDS.get("long_stay_rate_warning", 7)
        raw_kpis["long_stay_rate"] = mapping.long_stay_rate(los_threshold)
        
        # Benchmarks (ENGINE-AWARE, NOT DATA-DRIVEN)
        if raw_kpis.get("avg_los") is not None:
            raw_kpis["benchmark_los"] = HEALTHCARE_THRESHOLDS.get("benchmark_los", 5.0)
            raw_kpis["benchmark_source_los"] = "External Norm (Config)"
        
        # -------------------------------------------------
        # 4. Readmissions & quality (FACT)
        # -------------------------------------------------
        raw_kpis["readmission_rate"] = mapping.readmission_rate()
        
        # -------------------------------------------------
        # 5. Cost & financials (FACT)
        # -------------------------------------------------
        raw_kpis["total_billing"] = mapping.total_cost()
        raw_kpis["avg_cost_per_patient"] = mapping.avg_cost_per_patient()
        raw_kpis["avg_cost_per_day"] = mapping.avg_cost_per_day()
        
        # Cost benchmark anchoring (ENGINE RULE)
        if raw_kpis.get("avg_cost_per_patient") is not None:
            internal_benchmark = mapping.internal_cost_benchmark(multiplier=1.1)
            external_cap = HEALTHCARE_EXTERNAL_LIMITS.get(
                "avg_cost_per_patient", {}
            ).get("soft_cap", 50000)
            
            if internal_benchmark and internal_benchmark > external_cap:
                raw_kpis["benchmark_cost"] = external_cap
                raw_kpis["benchmark_source_cost"] = "External Cap"
            else:
                raw_kpis["benchmark_cost"] = internal_benchmark
                raw_kpis["benchmark_source_cost"] = "Internal Median (1.1x)"
        
        # -------------------------------------------------
        # 6. Variance & operational risk (FACT)
        # -------------------------------------------------
        raw_kpis["facility_variance_score"] = mapping.facility_variance()
        raw_kpis["provider_variance_score"] = mapping.provider_variance()
        
        # -------------------------------------------------
        # 7. Trends (FACT → SYMBOLIC)
        # -------------------------------------------------
        if self.time_col:
            raw_kpis["los_trend"] = mapping.trend(self.time_col, "los")
            raw_kpis["cost_trend"] = mapping.trend(self.time_col, "cost")
            raw_kpis["volume_trend"] = mapping.trend(self.time_col, "volume")
            
            if self.care_context == CareContext.INPATIENT:
                raw_kpis["weekend_admission_rate"] = mapping.weekend_admission_rate(self.time_col)
        
        # -------------------------------------------------
        # 8. Board confidence scoring (ENGINE)
        # -------------------------------------------------
        try:
            score, breakdown = _compute_board_confidence_score(raw_kpis, self.care_context)
        except Exception:
            score, breakdown = 50, {"Scoring Failure": -50}
        
        trend_symbol = (
            raw_kpis.get("los_trend")
            or raw_kpis.get("cost_trend")
            or "→"
        )
        
        band = (
            "Low" if score < 50
            else "Medium" if score < 70
            else "High" if score < 85
            else "Elite"
        )
        
        raw_kpis.update({
            "board_confidence_score": score,
            "board_confidence_band": band,
            "board_score_breakdown": breakdown,
            "board_confidence_trend": trend_symbol,
            "board_confidence_interpretation": _get_score_interpretation(score),
            "trend_explanation": _get_trend_explanation(trend_symbol),
            "maturity_level": _healthcare_maturity_level(score),
            "benchmark_context": "Benchmarks applied via engine governance rules",
        })
        
        # -------------------------------------------------
        # 9. Executive-safe KPI filtering (OUTPUT CONTRACT)
        # -------------------------------------------------
        # NOTE: Domain returns full KPI set.
        # Report layer MUST down-select to 3–5 executive KPIs.
        final_kpis = {}
        executive_order = [
            "board_confidence_score",
            "maturity_level",
            "total_patients",
            "avg_los",
            "long_stay_rate",
            "avg_cost_per_patient",
            "readmission_rate",
        ]
        
        for k in executive_order:
            if k in raw_kpis:
                final_kpis[k] = raw_kpis[k]
        
        for k, v in raw_kpis.items():
            if k not in final_kpis:
                if isinstance(v, Enum):
                    final_kpis[k] = v.value
                elif not (
                    isinstance(v, float)
                    and (np.isnan(v) or np.isinf(v))
                ):
                    final_kpis[k] = v
        
        return final_kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        
        def save(fig, name, caption, importance):
            fig.savefig(output_dir / name, bbox_inches="tight", dpi=100)
            plt.close(fig)
            visuals.append({
                "path": str(output_dir / name),
                "caption": caption,
                "importance": importance
            })
        
        def human_fmt(x, _):
            if x is None or pd.isna(x):
                return ""
            try:
                x = float(x)
            except Exception:
                return str(x)
            if abs(x) >= 1e6:
                return f"${x/1e6:.1f}M"
            if abs(x) >= 1e3:
                return f"${x/1e3:.0f}K"
            return f"{int(x)}"
        
        # 1️⃣ Activity Volume Trend
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                df.set_index(self.time_col).resample("ME").size().plot(ax=ax, linewidth=2)
                ax.set_title("Activity Volume Over Time", fontweight="bold")
                ax.grid(True, alpha=0.2)
                save(fig, "vol_trend.png", "Observed activity volume across time.", 0.99)
            except Exception:
                pass
        
        # 2️⃣ Duration / Cycle Time Distribution
        if c.get("los") and not df[c["los"]].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["los"]].hist(ax=ax, bins=15, alpha=0.7)
                ax.axvline(VISUAL_BENCHMARK_LOS, color='red', linestyle='--', linewidth=1.5, label=f'Goal ({VISUAL_BENCHMARK_LOS}d)')
                ax.legend()
                ax.set_title("Cycle Time Distribution", fontweight="bold")
                save(fig, "duration_dist.png", "Distribution of observed cycle durations.", 0.95)
            except Exception:
                pass
        
        # 3️⃣ Top Value Contributors
        if c.get("diagnosis") and c.get("cost"):
            try:
                stats = df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5)
                if not stats.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    stats.plot(kind="bar", ax=ax)
                    ax.set_title("Top Value Contributors", fontweight="bold")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    plt.xticks(rotation=45, ha="right")
                    save(fig, "value_drivers.png", "Categories with highest average value.", 0.92)
            except Exception:
                pass
        
        # 4️⃣ Risk / Failure Rate by Category
        if c.get("readmitted") and c.get("diagnosis") and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
            try:
                rates = df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5)
                if not rates.empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    rates.plot(kind="barh", ax=ax)
                    ax.set_title("Highest Risk Categories", fontweight="bold")
                    save(fig, "risk_rates.png", "Observed risk rates by category.", 0.90)
            except Exception:
                pass
        
        # 5️⃣ Value vs Duration Relationship
        if c.get("cost") and c.get("los") and c.get("diagnosis"):
            try:
                valid = df[[c["cost"], c["los"], c["diagnosis"]]].dropna()
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for key in valid[c["diagnosis"]].value_counts().nlargest(5).index:
                        subset = valid[valid[c["diagnosis"]] == key]
                        ax.scatter(subset[c["los"]], subset[c["cost"]], alpha=0.6, label=str(key))
                    ax.set_title("Value vs Duration", fontweight="bold")
                    ax.legend()
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "value_vs_duration.png", "Relationship between duration and value.", 0.88)
            except Exception:
                pass
        
        # 6️⃣ Operator / Provider Variance
        if c.get("doctor") and c.get("los"):
            try:
                count = df[c["doctor"]].nunique()
                if 3 <= count <= 30:
                    stats = df.groupby(c["doctor"])[c["los"]].mean().nlargest(10)
                    if not stats.empty:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        stats.plot(kind="bar", ax=ax)
                        ax.set_title("Operator-Level Variance", fontweight="bold")
                        plt.xticks(rotation=45, ha="right")
                        save(fig, "operator_variance.png", "Variation across operators.", 0.85)
            except Exception:
                pass
        
        # 7️⃣ Payer / Source Mix
        if c.get("payer"):
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct="%1.1f%%")
                ax.set_title("Source Mix")
                save(fig, "source_mix.png", "Distribution by source or payer.", 0.75)
            except Exception:
                pass
        
        # 8️⃣ Demographic Distribution
        if c.get("age"):
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["age"]].hist(ax=ax, bins=20, alpha=0.6)
                ax.set_title("Demographic Distribution")
                save(fig, "demographics.png", "Population age distribution.", 0.65)
            except Exception:
                pass
        
        # 9️⃣ Cost Distribution & Outliers
        if c.get("cost"):
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.boxplot(df[c["cost"]].dropna(), vert=False)
                ax.set_title("Value Distribution")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "value_box.png", "Distribution and outliers of value metric.", 0.60)
            except Exception:
                pass
        
        # 🔟 Category Type Mix
        if c.get("type"):
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["type"]].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
                ax.set_title("Category Mix")
                save(fig, "category_mix.png", "Composition by category type.", 0.55)
            except Exception:
                pass
        
        # 1️⃣1️⃣ Temporal Pattern (Day-of-Week)
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df["_dow"] = df[self.time_col].dt.day_name()
                df["_dow"].value_counts().reindex(
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                ).plot(kind="bar", ax=ax)
                ax.set_title("Temporal Pattern")
                save(fig, "temporal_pattern.png", "Activity distribution by weekday.", 0.50)
            except Exception:
                pass
        
        # 🚨 IMPORTANT: Domain generates MANY visuals
        # 🚨 Report layer decides how many to show
        return sorted(visuals, key=lambda x: x["importance"], reverse=True)
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any], shape_info=None):
        """
        UNIVERSAL INSIGHT GENERATOR (Domain-aware, Engine-neutral)

        Rules:
        - Insights explain KPIs, not raw data
        - No assumptions if required KPIs are missing
        - CRITICAL insights may be executive-summary eligible
        """

        insights: List[Dict[str, Any]] = []
        t = HEALTHCARE_THRESHOLDS
        c = self.cols

        # =================================================
        # 1. ROOT CAUSE: EXCESS LOS (DIAGNOSIS-LEVEL)
        # =================================================
        if (
            c.get("diagnosis")
            and c.get("los")
            and "avg_los" in kpis
            and not df[c["los"]].dropna().empty
        ):
            diag_perf = (
                df.groupby(c["diagnosis"])[c["los"]]
                .agg(mean="mean", count="count", std="std")
                .reset_index()
            )

            diag_perf = diag_perf[diag_perf["count"] > 5]
            diag_perf = diag_perf.sort_values("mean", ascending=False).head(3)

            drivers = []
            total_excess_days = 0.0
            global_target = kpis.get("benchmark_los", 5.0)

            for _, row in diag_perf.iterrows():
                if row["mean"] > global_target:
                    excess = row["mean"] - global_target
                    excess_days = excess * row["count"]
                    total_excess_days += excess_days

                    # Deterministic root cause classification
                    if row["std"] and row["std"] > (row["mean"] * 0.5):
                        cause_type = "Practice Variance"
                    elif row["mean"] > 10:
                        cause_type = "Clinical Complexity"
                    else:
                        cause_type = "Structural Delay"

                    drivers.append(
                        f"{row[c['diagnosis']]}: +{excess:.1f}d ({cause_type})"
                    )

            if drivers and total_excess_days > 0:
                avg_cost_per_day = kpis.get("avg_cost_per_day")
                if not avg_cost_per_day or avg_cost_per_day <= 0:
                    avg_cost_per_day = HEALTHCARE_EXTERNAL_LIMITS.get(
                        "avg_cost_per_day", 2000
                    )

                financial_impact = total_excess_days * avg_cost_per_day
                fin_str = (
                    f"${financial_impact/1e6:.1f}M"
                    if financial_impact >= 1e6
                    else f"${financial_impact/1e3:.0f}K"
                )

                blocked_beds = total_excess_days / 365

                insights.append({
                    "level": "CRITICAL",
                    "title": "Excess Length of Stay Drivers",
                    "so_what": (
                        f"Total excess stay days: {total_excess_days:,.0f} "
                        f"({fin_str} opportunity).<br/>"
                        f"Equivalent capacity blocked: {blocked_beds:.1f} beds.<br/>"
                        f"Primary drivers: " + "; ".join(drivers)
                    ),
                    "source": "LOS Root Cause Analysis",
                    "executive_summary_flag": True
                })

        # =================================================
        # 2. FACILITY VARIANCE
        # =================================================
        if (
            kpis.get("facility_variance_score", 0) > 0.5
            and c.get("facility")
            and c.get("los")
        ):
            fac_stats = df.groupby(c["facility"])[c["los"]].mean()

            if not fac_stats.empty and fac_stats.max() > fac_stats.min():
                best = fac_stats.idxmin()
                worst = fac_stats.idxmax()
                gap = fac_stats[worst] - fac_stats[best]

                insights.append({
                    "level": "RISK",
                    "title": "High Facility Performance Variance",
                    "so_what": (
                        f"Average LOS differs by {gap:.1f} days between facilities.<br/>"
                        f"Worst: {worst} ({fac_stats[worst]:.1f}d) vs "
                        f"Best: {best} ({fac_stats[best]:.1f}d)."
                    ),
                    "source": "Facility Comparison"
                })

        # =================================================
        # 3. QUALITY BLIND SPOT
        # =================================================
        if (
            self.care_context == CareContext.INPATIENT
            and "readmission_rate" in kpis
            and kpis.get("readmission_rate") is None
        ):
            insights.append({
                "level": "RISK",
                "title": "Quality Measurement Blind Spot",
                "so_what": (
                    "Readmission metrics are unavailable. Extended LOS cannot be "
                    "validated as quality-driven versus inefficiency-driven."
                ),
                "source": "Quality & Safety Controls"
            })

        # =================================================
        # 4. DISCHARGE BOTTLENECK
        # =================================================
        long_stay_rate = kpis.get("long_stay_rate")
        if (
            isinstance(long_stay_rate, (int, float))
            and long_stay_rate >= t.get("long_stay_rate_critical", 0.3)
        ):
            insights.append({
                "level": "CRITICAL",
                "title": "Severe Discharge Bottleneck",
                "so_what": (
                    f"{long_stay_rate:.1%} of patients exceed acceptable "
                    "length-of-stay thresholds."
                ),
                "source": "LOS Distribution Analysis",
                "executive_summary_flag": True
            })

        # =================================================
        # 5. COST ANOMALY
        # =================================================
        avg_cost = kpis.get("avg_cost_per_patient")
        benchmark_cost = kpis.get("benchmark_cost")

        if (
            isinstance(avg_cost, (int, float))
            and isinstance(benchmark_cost, (int, float))
            and avg_cost > benchmark_cost
        ):
            insights.append({
                "level": "WARNING",
                "title": "Cost Performance Anomaly",
                "so_what": (
                    "Average cost per case exceeds benchmark expectations, "
                    "indicating potential inefficiencies or pricing exposure."
                ),
                "source": "Financial Benchmarking"
            })

        # =================================================
        # 6. DATA QUALITY
        # =================================================
        data_completeness = kpis.get("data_completeness")
        if isinstance(data_completeness, (int, float)) and data_completeness < 0.90:
            insights.append({
                "level": "RISK",
                "title": "Data Integrity Gap",
                "so_what": (
                    "Key clinical or operational fields are missing, "
                    "reducing analytical precision and decision confidence."
                ),
                "source": "Data Quality Assessment"
            })

        # =================================================
        # 7. MINIMUM GUARANTEE PADDING
        # =================================================
        while len(insights) < 7:
            insights.append({
                "level": "INFO",
                "title": "Operational Observation",
                "so_what": "No additional statistically significant anomalies detected in this dataset.",
                "source": "System Generated"
            })

        return insights
    
    def generate_recommendations(self, df, kpis, insights=None, shape_info=None):
        recs = []
        titles = {i["title"] for i in (insights or [])}

        # Helper to add recommendations consistently
        def add(action, priority, owner, timeline, outcome, confidence, impact):
            recs.append({
                "action": action,
                "priority": priority,
                "owner": owner,
                "timeline": timeline,
                "expected_outcome": outcome,
                "confidence": round(float(confidence), 2),
                "impact_score": round(float(impact), 2)
            })

        # -------------------------------------------------
        # 1️⃣ Variance Reduction (Highest Universal ROI)
        # -------------------------------------------------
        if kpis.get("facility_variance_score", 0) > 0.5 or \
           kpis.get("provider_variance_score", 0) > 0.4:

            add(
                action="Reduce performance variance across entities",
                priority="HIGH",
                owner="Operations Leadership",
                timeline="60–90 days",
                outcome="Improved consistency on primary performance metric",
                confidence=0.85,
                impact=0.95
            )

        # -------------------------------------------------
        # 2️⃣ Bottleneck / Flow Constraint
        # -------------------------------------------------
        if "Severe Discharge Bottleneck" in titles or \
           kpis.get("long_stay_rate", 0) >= HEALTHCARE_THRESHOLDS.get("long_stay_rate_warning", 0.2):

            add(
                action="Analyze and remove primary process bottleneck",
                priority="HIGH",
                owner="Operational Excellence",
                timeline="30–60 days",
                outcome="Improved throughput and capacity utilization",
                confidence=0.90,
                impact=0.92
            )

        # -------------------------------------------------
        # 3️⃣ Root-Cause Hotspot Intervention
        # -------------------------------------------------
        # MATCHING TRIGGER TO ACTUAL INSIGHT TITLE (Fix 3)
        if "Excess Length of Stay Drivers" in titles:

            add(
                action="Apply targeted interventions to top-performing-impact categories",
                priority="HIGH",
                owner="Functional Leadership",
                timeline="90 days",
                outcome="Reduction in excess effort and cost concentration",
                confidence=0.88,
                impact=0.90
            )

        # -------------------------------------------------
        # 4️⃣ Missing Quality / Risk Signal
        # -------------------------------------------------
        # MATCHING TRIGGER TO ACTUAL INSIGHT TITLE
        if "Quality Measurement Blind Spot" in titles:

            add(
                action="Integrate missing quality or outcome indicators",
                priority="CRITICAL",
                owner="Data & Analytics",
                timeline="Immediate",
                outcome="Enable safety vs efficiency trade-off visibility",
                confidence=0.95,
                impact=0.93
            )

        # -------------------------------------------------
        # 5️⃣ Financial Outlier Control
        # -------------------------------------------------
        if kpis.get("avg_cost_per_patient", 0) > kpis.get("benchmark_cost", float("inf")):

            add(
                action="Investigate and control high-value outliers",
                priority="MEDIUM",
                owner="Finance & Operations",
                timeline="30–60 days",
                outcome="Improved unit economics and margin stability",
                confidence=0.80,
                impact=0.85
            )

        # -------------------------------------------------
        # 6️⃣ External Dependency Optimization (Optional)
        # -------------------------------------------------
        if self.cols.get("payer"):

            add(
                action="Review external partner or contract performance",
                priority="LOW",
                owner="Commercial / Revenue Teams",
                timeline="Annual",
                outcome="Improved alignment between cost and value realization",
                confidence=0.65,
                impact=0.60
            )

        # -------------------------------------------------
        # Fallback (No Critical Signals)
        # -------------------------------------------------
        if not recs:
            add(
                action="Continue monitoring key performance indicators",
                priority="LOW",
                owner="Operations",
                timeline="Ongoing",
                outcome="Sustained performance stability",
                confidence=0.60,
                impact=0.40
            )

        # -------------------------------------------------
        # 🔥 CRITICAL RULE
        # Sort by impact, NOT by priority label
        # Report layer will display top 3–5
        # -------------------------------------------------
        recs = sorted(recs, key=lambda x: x["impact_score"], reverse=True)

        # Remove internal-only field before returning
        for r in recs:
            r.pop("impact_score", None)

        return recs

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "medical_condition", "clinical", "doctor", "insurance", "encounter", "test_date"}
    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        return DomainDetectionResult("healthcare", min(len(hits)/3, 1.0), {})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
