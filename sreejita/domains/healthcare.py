import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
from matplotlib.ticker import FuncFormatter
from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# IMPORT THE TRUTH
from sreejita.narrative.benchmarks import HEALTHCARE_THRESHOLDS

# =====================================================
# CONSTANTS & STANDARDS
# =====================================================
VISUAL_BENCHMARK_LOS = 5.0

# FIX 1: Enhanced Care Settings (Diagnostic Support)
class CareSetting(str, Enum):
    INPATIENT = "inpatient"     # Has Admit + Discharge
    OUTPATIENT = "outpatient"   # Has Visit Date, No Overnight
    DIAGNOSTIC = "diagnostic"   # Lab/Test data (Encounter ID, no LOS)
    MIXED = "mixed"
    UNKNOWN = "unknown"

class DatasetShape(str, Enum):
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    UNKNOWN = "unknown"

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    score = {k: 0 for k in DatasetShape}

    if any(x in c for c in cols for x in ["patient", "mrn", "pid", "id", "name", "encounter", "visit"]): 
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
    if any(x in c for c in cols for x in ["admit", "admission", "date", "joining", "test_date"]): 
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
    
    if any(x in c for c in cols for x in ["total", "volume", "census", "visits"]): 
        score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
    
    if any(x in c for c in cols for x in ["revenue", "bill", "cost", "fee", "amount"]): 
        score[DatasetShape.FINANCIAL_SUMMARY] += 2

    best = max(score, key=score.get)
    if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 3: best = DatasetShape.ROW_LEVEL_CLINICAL
    if score[best] == 0: best = DatasetShape.UNKNOWN
    return {"shape": best, "score": score}

# =====================================================
# INTERNAL TREND ENGINE (FIX 3 - REAL TRENDS)
# =====================================================
def _calculate_internal_trend(df: pd.DataFrame, time_col: str, metric_col: str) -> str:
    """
    Compares the last 20% of the timeline vs the previous 80% 
    to determine a directional trend (↑, ↓, →) without external DB.
    """
    if not time_col or metric_col not in df.columns:
        return "→"
    
    try:
        # Sort by date
        df = df.sort_values(time_col).dropna(subset=[metric_col])
        if len(df) < 10: return "→"

        # Split timeline (Recent vs Historical)
        cutoff_idx = int(len(df) * 0.8)
        historical = df.iloc[:cutoff_idx][metric_col].mean()
        recent = df.iloc[cutoff_idx:][metric_col].mean()

        if historical == 0: return "→"
        delta = (recent - historical) / historical

        if delta > 0.05: return "↑"   # >5% increase
        if delta < -0.05: return "↓"  # >5% decrease
        return "→"
    except:
        return "→"

# =====================================================
# BOARD INTELLIGENCE LOGIC (ENTERPRISE)
# =====================================================
def _compute_board_confidence_score(kpis: Dict[str, Any]) -> int:
    """Calculates a 0-100 score representing executive confidence."""
    
    # --- AGGREGATED PATH ---
    if kpis.get("dataset_shape") == DatasetShape.AGGREGATED_OPERATIONAL:
        score = 70
        if kpis.get("avg_los") and kpis.get("benchmark_los"):
             if kpis["avg_los"] <= kpis["benchmark_los"]: score += 10
        if kpis.get("readmission_rate") and kpis["readmission_rate"] <= 0.1:
            score += 10
        return min(score, 85)

    # --- ROW LEVEL PATH ---
    score = 100
    t = HEALTHCARE_THRESHOLDS
    setting = kpis.get("care_setting", CareSetting.UNKNOWN)
    
    # 1. FACILITY PENALTY (Fix 2: Multi-Hospital Awareness)
    # If variance across hospitals is too high, reduce confidence in "Global" average
    if kpis.get("facility_variance_score", 0) > 0.5:
        score -= 10 

    # 2. DIAGNOSTIC / ENCOUNTER LOGIC (Fix 1: Lab Data Support)
    if setting == CareSetting.DIAGNOSTIC:
        # For labs, we care about Cost Variance & Volume Stability
        if kpis.get("cost_trend") == "↑": score -= 10
        if kpis.get("data_completeness", 0) < 0.9: score -= 20
        # No penalty for missing LOS
        return max(score, 0)

    # 3. TRADITIONAL HOSPITAL LOGIC
    # Operational
    if kpis.get("long_stay_rate", 0) >= t["long_stay_rate_critical"]: score -= 25
    elif kpis.get("long_stay_rate", 0) >= t["long_stay_rate_warning"]: score -= 15

    if kpis.get("avg_los") and kpis.get("benchmark_los"):
        if kpis["avg_los"] > kpis["benchmark_los"]: score -= 10

    # Clinical
    if kpis.get("readmission_rate", 0) >= t["readmission_critical"]: score -= 20
    elif kpis.get("readmission_rate", 0) >= t["readmission_warning"]: score -= 10

    # Financial
    if kpis.get("avg_cost_per_patient") and kpis.get("benchmark_cost"):
        if kpis["avg_cost_per_patient"] > kpis["benchmark_cost"] * t["cost_multiplier_warning"]: score -= 10

    # Missing Data Penalty (Context Aware)
    if setting == CareSetting.INPATIENT and kpis.get("avg_los") is None:
        score -= 15

    return max(score, 0)

def _healthcare_maturity_level(board_score: int) -> str:
    if board_score >= 85: return "Gold"
    if board_score >= 70: return "Silver"
    return "Bronze"

# =====================================================
# HEALTHCARE DOMAIN (FACT ENGINE)
# =====================================================
class HealthcareDomain(BaseDomain):
    name = "healthcare"
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.shape_info = detect_dataset_shape(df)
        self.shape = self.shape_info["shape"]
        
        # UNIVERSAL COLUMN MAPPING (Enhanced)
        self.cols = {
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "name") or resolve_column(df, "mrn"),
            "encounter": resolve_column(df, "encounter_id") or resolve_column(df, "visit_id") or resolve_column(df, "test_id"),
            "facility": resolve_column(df, "hospital_id") or resolve_column(df, "facility_code") or resolve_column(df, "location") or resolve_column(df, "center"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition") or resolve_column(df, "condition"),
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost") or resolve_column(df, "charges"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "admission_type") or resolve_column(df, "readmission"),
            "volume": resolve_column(df, "total_patients") or resolve_column(df, "visits"),
            "avg_los": resolve_column(df, "avg_los"),
            "admit": resolve_column(df, "date_of_admission") or resolve_column(df, "admission_date") or resolve_column(df, "joining_date") or resolve_column(df, "test_date"),
            "discharge": resolve_column(df, "discharge_date") or resolve_column(df, "date_of_discharge"),
            "payer": resolve_column(df, "insurance_provider") or resolve_column(df, "payer") or resolve_column(df, "insurance"),
            "age": resolve_column(df, "age") or resolve_column(df, "dob"),
            "type": resolve_column(df, "admission_type") or resolve_column(df, "type"),
        }

        # FIX 1: ENHANCED CARE SETTING DETECTION
        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL:
            if self.cols["admit"] and self.cols["discharge"]:
                self.care_setting = CareSetting.INPATIENT
            elif self.cols["encounter"] and not self.cols["los"] and not self.cols["discharge"]:
                # Has Encounter ID but no stay duration -> Diagnostic/Lab
                self.care_setting = CareSetting.DIAGNOSTIC
            elif not self.cols["los"] and not self.cols["discharge"]:
                self.care_setting = CareSetting.OUTPATIENT
            else:
                self.care_setting = CareSetting.MIXED
        else:
            self.care_setting = CareSetting.UNKNOWN
        
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

        # Time Column Setup
        if self.cols["admit"]:
            df[self.cols["admit"]] = pd.to_datetime(df[self.cols["admit"]], errors="coerce")
            self.time_col = self.cols["admit"]
            df = df.sort_values(self.time_col)
        else: self.time_col = None

        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        raw_kpis = {
            "dataset_shape": self.shape, 
            "care_setting": self.care_setting
        }
        c = self.cols
        
        valid_cols = [v for v in c.values() if v]
        raw_kpis["data_completeness"] = round(1 - df[valid_cols].isna().mean().mean(), 2) if valid_cols else 0.0

        # --- AGGREGATED PATH ---
        if self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            if c["volume"]: raw_kpis["total_patients"] = df[c["volume"]].sum()
            if c["avg_los"]: 
                raw_kpis["avg_los"] = df[c["avg_los"]].mean()
                raw_kpis["benchmark_los"] = 5.0 
            raw_kpis["is_aggregated"] = True
            
        # --- ROW LEVEL PATH ---
        else:
            raw_kpis["is_aggregated"] = False
            raw_kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)
            
            # 1. ENCOUNTER / DIAGNOSTIC SPECIFIC
            if self.care_setting == CareSetting.DIAGNOSTIC:
                raw_kpis["total_encounters"] = len(df)
                if c["pid"]:
                    raw_kpis["visits_per_patient"] = len(df) / raw_kpis["total_patients"]
            
            # 2. INPATIENT SPECIFIC
            else:
                if c["los"] and not df[c["los"]].dropna().empty:
                    raw_kpis["avg_los"] = df[c["los"]].mean()
                    raw_kpis["long_stay_rate"] = (df[c["los"]] > 7).mean()
                    raw_kpis["benchmark_los"] = 5.0
                else: raw_kpis["avg_los"] = None

                if c["readmitted"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
                    raw_kpis["readmission_rate"] = df[c["readmitted"]].mean()

            # 3. UNIVERSAL FINANCIALS
            if c["cost"]:
                raw_kpis["total_billing"] = df[c["cost"]].sum()
                raw_kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
                raw_kpis["benchmark_cost"] = df[c["cost"]].median() * 2.0
                if raw_kpis.get("avg_los"): 
                    raw_kpis["avg_cost_per_day"] = raw_kpis["avg_cost_per_patient"] / raw_kpis["avg_los"]

            # 4. FIX 2: MULTI-FACILITY INTELLIGENCE
            if c["facility"]:
                raw_kpis["facility_count"] = df[c["facility"]].nunique()
                # Compute Variance across facilities (Coefficient of Variation)
                if c["cost"]:
                    fac_stats = df.groupby(c["facility"])[c["cost"]].mean()
                    if fac_stats.mean() > 0:
                        raw_kpis["facility_variance_score"] = fac_stats.std() / fac_stats.mean()
            
            # 5. FIX 3: REAL TRENDS (Internal)
            if self.time_col:
                if c["cost"]: 
                    raw_kpis["cost_trend"] = _calculate_internal_trend(df, self.time_col, c["cost"])
                if c["volume"]: # If pre-aggregated
                    raw_kpis["vol_trend"] = _calculate_internal_trend(df, self.time_col, c["volume"])

            # Clinical Variance
            if c["doctor"] and c["los"]:
                stats = df.groupby(c["doctor"])[c["los"]].mean()
                if stats.mean() > 0: raw_kpis["provider_variance_score"] = stats.std() / stats.mean()

        # 🔥 BOARD INTELLIGENCE
        current_score = _compute_board_confidence_score(raw_kpis)
        raw_kpis["board_confidence_score"] = current_score
        raw_kpis["maturity_level"] = _healthcare_maturity_level(current_score)
        
        # Link trend to Board Score Trend (using internal cost trend as proxy if available)
        raw_kpis["board_confidence_trend"] = raw_kpis.get("cost_trend", "→")

        # Filter out Enums for cleaner JSON
        final_kpis = {}
        for k, v in raw_kpis.items():
            if isinstance(v, Enum):
                final_kpis[k] = v.value
            elif not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                final_kpis[k] = v
                
        return final_kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        # ... (Same visualization logic as before) ...
        # I am omitting the full visual code block for brevity since it didn't change logic,
        # but you should keep the existing generate_visuals function here.
        # Just ensure you use self.cols correctly.
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        def save(fig, name, cap, imp):
            fig.savefig(output_dir / name, bbox_inches="tight")
            plt.close(fig)
            visuals.append({"path": str(output_dir / name), "caption": cap, "importance": imp})
        def human_fmt(x, _):
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        # 1. Volume
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                df.set_index(self.time_col).resample('ME').size().plot(ax=ax, color="#1f77b4")
                ax.set_title("Patient Volume Trend")
                save(fig, "vol.png", "Demand stability", 0.99)
            except: pass

        # 2. LOS Distribution (Benchmark)
        if c["los"] and not df[c["los"]].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["los"]].hist(ax=ax, bins=15, color="teal", alpha=0.7)
                ax.axvline(VISUAL_BENCHMARK_LOS, color='red', linestyle='--', linewidth=1.5, label=f'Goal ({VISUAL_BENCHMARK_LOS}d)')
                ax.legend()
                ax.set_title("LOS Distribution vs Goal")
                save(fig, "los.png", "Stay duration & adherence", 0.95)
            except: pass

        # 3. Cost by Condition
        if c["diagnosis"] and c["cost"]:
            try:
                if not df[c["cost"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot(kind="bar", ax=ax, color="orange")
                    ax.set_title("Top Cost Drivers")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost.png", "Cost drivers", 0.90)
            except: pass

        # 4. Readmission
        if c["readmitted"] and c["diagnosis"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
            try:
                if not df[c["readmitted"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax, color="red")
                    ax.set_title("Readmission Risk Areas")
                    save(fig, "readm.png", "Clinical risk", 0.88)
            except: pass

        # 5. Cost vs LOS
        if c["cost"] and c["los"]:
            try:
                valid = df[[c["cost"], c["los"]]].dropna()
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(valid[c["los"]], valid[c["cost"]], alpha=0.5, color="gray", s=15)
                    ax.set_title("Cost vs. LOS Correlation")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost_los.png", "Efficiency", 0.85)
            except: pass

        # 6. Provider Variance
        if c["doctor"] and c["los"]:
            try:
                if not df[c["los"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot(kind="bar", ax=ax, color="brown")
                    ax.set_title("Provider Variance (Avg LOS)")
                    save(fig, "prov.png", "Care consistency", 0.80)
            except: pass

        # 7. Payer Mix
        if c["payer"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
                save(fig, "payer.png", "Revenue source", 0.75)
            except: pass

        # 8. Age
        if c["age"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["age"]].hist(ax=ax, bins=20, color="green", alpha=0.6)
                ax.set_title("Patient Demographics")
                save(fig, "age.png", "Demographics", 0.60)
            except: pass

        # 9. Cost Boxplot
        if c["cost"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.boxplot(df[c["cost"]].dropna(), vert=False)
                ax.set_title("Cost Outliers")
                save(fig, "cost_box.png", "Financial outliers", 0.55)
            except: pass

        # 10. Admission Type
        if c["type"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["type"]].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%')
                ax.set_title("Admission Types")
                save(fig, "type.png", "Acuity mix", 0.50)
            except: pass

        # 11. Day of Week
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df["_dow"] = df[self.time_col].dt.dayofweek
                df["_dow"].value_counts().sort_index().plot(kind="bar", ax=ax, color="purple")
                ax.set_title("Admissions by Day")
                save(fig, "dow.png", "Staffing", 0.45)
            except: pass

        return sorted(visuals, key=lambda x: x["importance"], reverse=True)[:6]

    def generate_insights(self, df, kpis, shape_info=None):
        insights = []
        t = HEALTHCARE_THRESHOLDS
        
        # FIX 2: Multi-Facility Insight
        if kpis.get("facility_variance_score", 0) > 0.5:
            insights.append({"level": "RISK", "title": "High Facility Variance", "so_what": "Significant cost/quality gaps exist between hospital sites."})

        # FIX 1: Diagnostic Insight
        if kpis.get("care_setting") == "diagnostic":
            insights.append({"level": "INFO", "title": "Diagnostic Dataset", "so_what": f"Analysis focused on {kpis.get('total_encounters',0)} clinical encounters."})

        # 1. Existing Insights
        if isinstance(kpis.get("long_stay_rate"), (int, float)) and kpis["long_stay_rate"] >= t["long_stay_rate_critical"]:
            insights.append({"level": "CRITICAL", "title": "Severe Discharge Bottleneck", "so_what": f"{kpis['long_stay_rate']:.1%} of patients exceed targets."})

        # 2. Systemic Inefficiency
        if (kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"] and kpis.get("readmission_rate", 0) >= t["readmission_critical"]):
            insights.append({"level": "CRITICAL", "title": "Systemic Care Inefficiency", "so_what": "Extended stays + high readmissions."})
        
        # 3. Efficiency Warning
        if kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"] and not any(i["title"] == "Severe Discharge Bottleneck" for i in insights):
            insights.append({"level": "WARNING", "title": "Extended Inpatient Stay", "so_what": f"Avg LOS ({kpis['avg_los']:.1f} days) exceeds benchmarks."})
        
        # 4. Staffing Stress
        if kpis.get("weekend_admission_rate", 0) >= t["weekend_rate_warning"]:
            insights.append({"level": "WARNING", "title": "Weekend Capacity Stress", "so_what": "Weekend volume strains staffing."})
        
        # 5. Clinical Variance
        if kpis.get("provider_variance_score", 0) >= t["provider_variance_warning"]:
            insights.append({"level": "RISK", "title": "High Provider Variance", "so_what": "Significant variability in LOS across providers."})
        
        # 6. Financial Risk
        if kpis.get("avg_cost_per_patient", 0) > kpis.get("benchmark_cost", 999999) * 1.5:
            insights.append({"level": "WARNING", "title": "Cost Anomaly", "so_what": "Costs significantly exceed benchmarks."})
            
        if kpis.get("data_completeness", 1) < 0.90:
            insights.append({"level": "RISK", "title": "Data Integrity Gap", "so_what": "Missing clinical fields limit precision."})

        if not insights:
            insights.append({"level": "INFO", "title": "Stable Operations", "so_what": "Metrics within tolerance."})
        return insights

    def generate_recommendations(self, df, kpis, insights=None, shape_info=None):
        recs = []
        titles = [i["title"] for i in (insights or [])]
        
        if "High Facility Variance" in titles:
             recs.append({"action": "Standardize protocols across sites", "priority": "HIGH", "timeline": "Q2", "owner": "Ops Director", "expected_outcome": "Unified Care"})

        if "Severe Discharge Bottleneck" in titles:
            recs.append({"action": "Audit discharge planning", "priority": "HIGH", "timeline": "30 days", "owner": "Clinical Ops", "expected_outcome": "Reduce LOS"})

        if kpis.get("readmission_rate", 0) > t["readmission_warning"]:
            recs.append({"action": "Implement post-discharge calls", "priority": "HIGH", "timeline": "Immediate", "owner": "Nursing", "expected_outcome": "Reduce Readmits"})
        
        if "Weekend Capacity Stress" in titles:
            recs.append({"action": "Adjust weekend staffing", "priority": "MEDIUM", "timeline": "Next Quarter", "owner": "HR", "expected_outcome": "Capacity Balance"})
        
        if "High Provider Variance" in titles:
            recs.append({"action": "Standardize treatment protocols", "priority": "MEDIUM", "timeline": "90 days", "owner": "CMO", "expected_outcome": "Reduce Variance"})
        
        if kpis.get("avg_cost_per_patient", 0) > 50000:
            recs.append({"action": "Review high-cost outliers", "priority": "MEDIUM", "timeline": "30 days", "owner": "Finance", "expected_outcome": "Recover Revenue"})
        
        if self.cols["payer"]:
            recs.append({"action": "Evaluate payer contracts", "priority": "LOW", "timeline": "Annual", "owner": "Revenue Cycle", "expected_outcome": "Optimize Yield"})

        if kpis.get("data_completeness", 1) < 0.9:
            recs.append({"action": "Validate ETL timestamps", "priority": "HIGH", "timeline": "Immediate", "owner": "IT", "expected_outcome": "Enable Analytics"})
            
        if not recs:
            recs.append({"action": "Monitor trends", "priority": "LOW", "timeline": "Ongoing", "owner": "Ops", "expected_outcome": "Stability"})
        
        return recs[:5]

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "medical_condition", "clinical", "doctor", "insurance", "encounter", "test_date"}
    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        return DomainDetectionResult("healthcare", min(len(hits)/3, 1.0), {})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
