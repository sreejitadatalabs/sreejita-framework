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

# IMPORT THE TRUTH (Ensure benchmarks.py is in sreejita/narrative/)
from sreejita.narrative.benchmarks import HEALTHCARE_THRESHOLDS

# =====================================================
# CONSTANTS & STANDARDS
# =====================================================
VISUAL_BENCHMARK_LOS = 5.0

# 1️⃣ EXPLICIT CARE-CONTEXT RESOLUTION
class CareContext(str, Enum):
    INPATIENT = "inpatient"     # Hospitals (Has LOS)
    OUTPATIENT = "outpatient"   # Clinics (Visits, No Bed)
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
    CareContext.DIAGNOSTIC: {"total_encounters", "data_completeness"},
    CareContext.AGGREGATED: {}, 
    CareContext.UNKNOWN: {}
}

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

def _calculate_internal_trend(df: pd.DataFrame, time_col: str, metric_col: str) -> str:
    """Compares recent vs historical data for real trends (↑, ↓, →)."""
    if not time_col or metric_col not in df.columns: return "→"
    try:
        df = df.sort_values(time_col).dropna(subset=[metric_col])
        if len(df) < 10: return "→"
        cutoff_idx = int(len(df) * 0.8)
        historical = df.iloc[:cutoff_idx][metric_col].mean()
        recent = df.iloc[cutoff_idx:][metric_col].mean()
        if historical == 0: return "→"
        delta = (recent - historical) / historical
        if delta > 0.05: return "↑"
        if delta < -0.05: return "↓"
        return "→"
    except: return "→"

# =====================================================
# 3️⃣ BOARD CONFIDENCE (Weighted & Explained)
# =====================================================
def _compute_board_confidence_score(kpis: Dict[str, Any], context: CareContext) -> Tuple[int, List[str]]:
    score = 100
    explanation = []
    t = HEALTHCARE_THRESHOLDS
    
    explanation.append(f"Context detected as **{context.value.upper()}** based on data shape.")

    # --- AGGREGATED PATH ---
    if context == CareContext.AGGREGATED:
        score = 70
        explanation.append("Baseline confidence (70/100) applied for aggregated summary data.")
        if kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] <= kpis["benchmark_los"]: 
            score += 10
            explanation.append("Bonus (+10): Avg LOS is within target.")
        if kpis.get("readmission_rate") and kpis["readmission_rate"] <= 0.1:
            score += 10
            explanation.append("Bonus (+10): Readmission rate is optimal.")
        final_score = min(score, 85)
        if final_score < score: explanation.append("Score capped at 85/100 due to lack of granular row-level data.")
        return final_score, explanation

    # --- ROW LEVEL PATH ---
    # A. Check EXPECTED metrics
    required = EXPECTED_METRICS.get(context, set())
    for metric in required:
        if kpis.get(metric) is None:
            score -= 10
            explanation.append(f"Penalty (-10): Expected metric '{metric}' is missing from this {context.value} dataset.")

    # B. Facility Variance
    if kpis.get("facility_variance_score", 0) > 0.5:
        score -= 10
        explanation.append("Penalty (-10): High performance variance detected across facility locations.")

    # C. Threshold Penalties
    if kpis.get("long_stay_rate", 0) >= t["long_stay_rate_critical"]: 
        score -= 25
        explanation.append("Penalty (-25): Critical bottleneck in long-stay discharges.")
    elif kpis.get("long_stay_rate", 0) >= t["long_stay_rate_warning"]: 
        score -= 15
        explanation.append("Penalty (-15): Long-stay rate indicates emerging bottleneck.")

    if kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"]: 
        score -= 10
        explanation.append("Penalty (-10): Average LOS exceeds benchmark.")

    if kpis.get("readmission_rate", 0) >= t["readmission_critical"]: 
        score -= 20
        explanation.append("Penalty (-20): Readmission rate is critically high.")
    
    if kpis.get("avg_cost_per_patient") and kpis.get("benchmark_cost"):
        if kpis["avg_cost_per_patient"] > kpis["benchmark_cost"] * t["cost_multiplier_warning"]: 
            score -= 10
            explanation.append("Penalty (-10): Cost per patient significantly exceeds internal baseline.")

    # D. Diagnostic Checks
    if context == CareContext.DIAGNOSTIC:
        if kpis.get("data_completeness", 0) < 0.9: 
            score -= 20
            explanation.append("Penalty (-20): Diagnostic data completeness is below 90%.")

    if score == 100:
        explanation.append("Perfect Score: All metrics within optimal operational thresholds.")

    return max(score, 0), explanation

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
        
        # UNIVERSAL COLUMN MAPPING
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

        # CONTEXT RESOLUTION
        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL:
            if self.cols["los"] or (self.cols["admit"] and self.cols["discharge"]):
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
        raw_kpis = {
            "dataset_shape": self.shape, 
            "care_context": self.care_context
        }
        c = self.cols
        
        valid_cols = [v for v in c.values() if v]
        raw_kpis["data_completeness"] = round(1 - df[valid_cols].isna().mean().mean(), 2) if valid_cols else 0.0

        if self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            if c["volume"]: raw_kpis["total_patients"] = df[c["volume"]].sum()
            if c["avg_los"]: 
                raw_kpis["avg_los"] = df[c["avg_los"]].mean()
                raw_kpis["benchmark_los"] = 5.0 
            raw_kpis["is_aggregated"] = True
        else:
            raw_kpis["is_aggregated"] = False
            raw_kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)
            
            if self.care_context == CareContext.DIAGNOSTIC:
                raw_kpis["total_encounters"] = len(df)
                if c["pid"]: raw_kpis["visits_per_patient"] = len(df) / raw_kpis["total_patients"]
            else:
                if c["los"] and not df[c["los"]].dropna().empty:
                    raw_kpis["avg_los"] = df[c["los"]].mean()
                    raw_kpis["long_stay_rate"] = (df[c["los"]] > 7).mean()
                    raw_kpis["benchmark_los"] = 5.0
                else: raw_kpis["avg_los"] = None
                if c["readmitted"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
                    raw_kpis["readmission_rate"] = df[c["readmitted"]].mean()

            if c["cost"]:
                raw_kpis["total_billing"] = df[c["cost"]].sum()
                raw_kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
                raw_kpis["benchmark_cost"] = df[c["cost"]].median() * 2.0
                if raw_kpis.get("avg_los"): 
                    raw_kpis["avg_cost_per_day"] = raw_kpis["avg_cost_per_patient"] / raw_kpis["avg_los"]

            if c["facility"]:
                raw_kpis["facility_count"] = df[c["facility"]].nunique()
                if c["cost"]:
                    fac_stats = df.groupby(c["facility"])[c["cost"]].mean()
                    if fac_stats.mean() > 0:
                        raw_kpis["facility_variance_score"] = fac_stats.std() / fac_stats.mean()
            
            # 🔥 REAL TRENDS
            if self.time_col:
                if c["cost"]: raw_kpis["cost_trend"] = _calculate_internal_trend(df, self.time_col, c["cost"])
                if c["volume"]: raw_kpis["vol_trend"] = _calculate_internal_trend(df, self.time_col, c["volume"])
                if self.care_context == CareContext.INPATIENT:
                     df["_dow"] = df[self.time_col].dt.dayofweek
                     raw_kpis["weekend_admission_rate"] = df["_dow"].isin([5, 6]).mean()

            if c["doctor"] and c["los"]:
                stats = df.groupby(c["doctor"])[c["los"]].mean()
                if stats.mean() > 0: raw_kpis["provider_variance_score"] = stats.std() / stats.mean()

        # 🔥 BOARD INTELLIGENCE
        current_score, explanation_list = _compute_board_confidence_score(raw_kpis, self.care_context)
        raw_kpis["board_confidence_score"] = current_score
        raw_kpis["board_confidence_explanation"] = explanation_list
        raw_kpis["maturity_level"] = _healthcare_maturity_level(current_score)
        raw_kpis["board_confidence_trend"] = raw_kpis.get("cost_trend", "→")

        # PRIORITY SORTING (Most important KPIs first)
        ordered_keys = ["board_confidence_score", "maturity_level", "total_patients", "avg_cost_per_patient", "avg_los", "readmission_rate", "long_stay_rate"]
        final_kpis = {k: raw_kpis[k] for k in ordered_keys if k in raw_kpis and not (isinstance(raw_kpis[k], float) and (np.isnan(raw_kpis[k]) or np.isinf(raw_kpis[k])))}
        
        # Add the rest
        for k, v in raw_kpis.items():
            if k not in final_kpis:
                if isinstance(v, Enum): final_kpis[k] = v.value
                elif not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): final_kpis[k] = v
        
        return final_kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        
        def save(fig, name, cap, imp):
            fig.savefig(output_dir / name, bbox_inches="tight", dpi=100)
            plt.close(fig)
            visuals.append({"path": str(output_dir / name), "caption": cap, "importance": imp})
        
        def human_fmt(x, _):
            if x >= 1e6: return f"${x/1e6:.1f}M"
            if x >= 1e3: return f"${x/1e3:.0f}K"
            return str(int(x))

        # 1. Volume
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                df.set_index(self.time_col).resample('ME').size().plot(ax=ax, color="#1f77b4", linewidth=2)
                ax.set_title("Patient Volume Trend", fontweight='bold')
                ax.grid(True, alpha=0.2)
                save(fig, "vol.png", "Demand stability over time", 0.99)
            except: pass

        # 2. LOS
        if c["los"] and not df[c["los"]].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["los"]].hist(ax=ax, bins=15, color="teal", alpha=0.7)
                ax.axvline(VISUAL_BENCHMARK_LOS, color='red', linestyle='--', linewidth=1.5, label=f'Goal ({VISUAL_BENCHMARK_LOS}d)')
                ax.legend()
                ax.set_title("LOS Distribution vs Goal", fontweight='bold')
                save(fig, "los.png", "Stay duration & adherence", 0.95)
            except: pass

        # 3. Cost Drivers
        if c["diagnosis"] and c["cost"]:
            try:
                if not df[c["cost"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    stats = df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5)
                    colors = ['#d62728' if i == 0 else '#ff7f0e' for i in range(len(stats))]
                    stats.plot(kind="bar", ax=ax, color=colors)
                    top_condition = stats.idxmax()
                    ax.set_title(f"Highest Cost Driver: {top_condition}", fontweight='bold')
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    plt.xticks(rotation=45, ha='right')
                    save(fig, "cost.png", "Cost drivers by condition", 0.90)
            except: pass

        # 4. Readmission
        if c["readmitted"] and c["diagnosis"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
            try:
                if not df[c["readmitted"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax, color="#d62728", alpha=0.8)
                    ax.set_title("Highest Readmission Risks", fontweight='bold')
                    save(fig, "readm.png", "Clinical risk areas", 0.88)
            except: pass

        # 5. Cost vs LOS
        if c["cost"] and c["los"] and c["diagnosis"]:
            try:
                valid = df[[c["cost"], c["los"], c["diagnosis"]]].dropna()
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    top_diag = valid[c["diagnosis"]].value_counts().nlargest(5).index
                    for diag in top_diag:
                        subset = valid[valid[c["diagnosis"]] == diag]
                        ax.scatter(subset[c["los"]], subset[c["cost"]], label=diag, alpha=0.6, s=40)
                    ax.set_title("Cost vs. LOS: Clinical Correlation", fontweight='bold')
                    ax.legend(title="Condition")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost_los.png", "Efficiency outliers", 0.85)
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
                df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
                save(fig, "payer.png", "Revenue source mix", 0.75)
            except: pass

        # 8. Age
        if c["age"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["age"]].hist(ax=ax, bins=20, color="green", alpha=0.6)
                ax.set_title("Patient Age Distribution")
                save(fig, "age.png", "Demographics", 0.60)
            except: pass

        # 9. Cost Boxplot
        if c["cost"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.boxplot(df[c["cost"]].dropna(), vert=False)
                ax.set_title("Cost Distribution & Outliers")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "cost_box.png", "Financial outliers", 0.55)
            except: pass

        # 10. Admission Type
        if c["type"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["type"]].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_title("Admission Types")
                save(fig, "type.png", "Acuity mix", 0.50)
            except: pass

        # 11. Weekday Analysis
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df["_dow"] = df[self.time_col].dt.day_name()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df["_dow"] = pd.Categorical(df["_dow"], categories=days, ordered=True)
                df["_dow"].value_counts().sort_index().plot(kind="bar", ax=ax, color="purple", alpha=0.7)
                ax.set_title("Admissions by Day of Week")
                save(fig, "dow.png", "Staffing alignment", 0.45)
            except: pass
                
        # MAX VISUALS FOR REPORT = 6
        return sorted(visuals, key=lambda x: x["importance"], reverse=True)[:6]

    def generate_insights(self, df, kpis, shape_info=None):
        insights = []
        t = HEALTHCARE_THRESHOLDS
        
        # 1. Facility Variance
        if kpis.get("facility_variance_score", 0) > 0.5:
            insights.append({"level": "RISK", "title": "High Facility Variance", "so_what": "Significant cost/quality gaps exist between hospital sites."})

        # 2. Diagnostic Context
        if kpis.get("care_context") == "diagnostic":
            insights.append({"level": "INFO", "title": "Diagnostic Dataset", "so_what": f"Analysis focused on {kpis.get('total_encounters',0)} clinical encounters."})

        # 3. Discharge Bottleneck
        if isinstance(kpis.get("long_stay_rate"), (int, float)) and kpis["long_stay_rate"] >= t["long_stay_rate_critical"]:
            insights.append({"level": "CRITICAL", "title": "Severe Discharge Bottleneck", "so_what": f"{kpis['long_stay_rate']:.1%} of patients exceed targets."})

        # 4. Systemic Inefficiency
        if (kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"] and kpis.get("readmission_rate", 0) >= t["readmission_critical"]):
            insights.append({"level": "CRITICAL", "title": "Systemic Care Inefficiency", "so_what": "Extended stays + high readmissions."})

        # 5. Weekend Stress
        if kpis.get("weekend_admission_rate", 0) >= t.get("weekend_rate_warning", 0.35):
            insights.append({"level": "WARNING", "title": "Weekend Capacity Stress", "so_what": "Weekend volume strains staffing."})

        # 6. Cost Anomaly
        if kpis.get("avg_cost_per_patient", 0) > kpis.get("benchmark_cost", 999999) * 1.5:
            insights.append({"level": "WARNING", "title": "Cost Anomaly", "so_what": "Costs significantly exceed benchmarks."})

        # 7. Data Risk
        if kpis.get("data_completeness", 1) < 0.90:
            insights.append({"level": "RISK", "title": "Data Integrity Gap", "so_what": "Missing clinical fields limit precision."})

        if not insights:
            insights.append({"level": "INFO", "title": "Stable Operations", "so_what": "Metrics within tolerance."})
        return insights

    def generate_recommendations(self, df, kpis, insights=None, shape_info=None):
        recs = []
        t = HEALTHCARE_THRESHOLDS
        titles = [i["title"] for i in (insights or [])]
        
        if "High Facility Variance" in titles:
             recs.append({"action": "Standardize protocols across sites", "priority": "HIGH", "timeline": "Q2", "owner": "Ops Director", "expected_outcome": "Unified Care"})

        if "Severe Discharge Bottleneck" in titles:
            recs.append({"action": "Audit discharge planning", "priority": "HIGH", "timeline": "30 days", "owner": "Clinical Ops", "expected_outcome": "Reduce LOS"})

        if kpis.get("readmission_rate", 0) > t["readmission_warning"]:
            recs.append({"action": "Implement post-discharge calls", "priority": "HIGH", "timeline": "Immediate", "owner": "Nursing", "expected_outcome": "Reduce Readmits"})
        
        if "Weekend Capacity Stress" in titles:
            recs.append({"action": "Adjust weekend staffing", "priority": "MEDIUM", "timeline": "Next Quarter", "owner": "HR", "expected_outcome": "Capacity Balance"})

        if "High Provider Variance" in titles or kpis.get("provider_variance_score", 0) > 0.4:
            recs.append({"action": "Standardize treatment protocols", "priority": "MEDIUM", "timeline": "90 days", "owner": "CMO", "expected_outcome": "Reduce Variance"})
        
        if kpis.get("avg_cost_per_patient", 0) > 50000 or "Cost Anomaly" in titles:
            recs.append({"action": "Review high-cost outliers", "priority": "MEDIUM", "timeline": "30 days", "owner": "Finance", "expected_outcome": "Recover Revenue"})
        
        if self.cols["payer"]:
            recs.append({"action": "Evaluate payer contracts", "priority": "LOW", "timeline": "Annual", "owner": "Revenue Cycle", "expected_outcome": "Optimize Yield"})

        if not recs:
            recs.append({"action": "Monitor trends", "priority": "LOW", "timeline": "Ongoing", "owner": "Ops", "expected_outcome": "Stability"})
        
        # MINIMUM 3, RETURN 5 FOR REPORT
        return recs[:5]

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "medical_condition", "clinical", "doctor", "insurance", "encounter", "test_date"}
    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        return DomainDetectionResult("healthcare", min(len(hits)/3, 1.0), {})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
