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
from sreejita.narrative.benchmarks import HEALTHCARE_THRESHOLDS as T

# =====================================================
# CONSTANTS
# =====================================================
VISUAL_BENCHMARK_LOS = 5.0

# =====================================================
# SHAPE DETECTION
# =====================================================
class DatasetShape(str, Enum):
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    QUALITY_METRICS = "quality_metrics"
    UNKNOWN = "unknown"

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    score = {k: 0 for k in DatasetShape}
    if any(x in c for c in cols for x in ["patient", "mrn", "pid", "id"]): score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
    if any(x in c for c in cols for x in ["admit", "admission", "date"]): score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
    if any(x in c for c in cols for x in ["total", "volume", "census"]): score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
    if any(x in c for c in cols for x in ["revenue", "bill", "cost"]): score[DatasetShape.FINANCIAL_SUMMARY] += 2
    best = max(score, key=score.get)
    if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 3: best = DatasetShape.ROW_LEVEL_CLINICAL
    if score[best] == 0: best = DatasetShape.UNKNOWN
    return {"shape": best, "score": score}

# =====================================================
# BOARD INTELLIGENCE LOGIC
# =====================================================
def _compute_board_confidence_score(kpis: Dict[str, Any]) -> int:
    score = 100
    if kpis.get("long_stay_rate", 0) >= T["long_stay_rate_critical"]: score -= 25
    elif kpis.get("long_stay_rate", 0) >= T["long_stay_rate_warning"]: score -= 15
    
    if kpis.get("avg_los") and kpis.get("benchmark_los"):
        if kpis["avg_los"] > kpis["benchmark_los"]: score -= 10

    if kpis.get("readmission_rate", 0) >= T["readmission_critical"]: score -= 20
    elif kpis.get("readmission_rate", 0) >= T["readmission_warning"]: score -= 10

    if kpis.get("avg_cost_per_patient") and kpis.get("benchmark_cost"):
        if kpis["avg_cost_per_patient"] > kpis["benchmark_cost"] * T["cost_multiplier_warning"]: score -= 10

    if kpis.get("avg_los") is None: score -= 15
    return max(score, 0)

def _healthcare_maturity_level(board_score: int) -> str:
    if board_score >= 85: return "Gold"
    if board_score >= 70: return "Silver"
    return "Bronze"

def _trend_delta(current: int, previous: Optional[int]) -> str:
    if previous is None: return "→"
    if current >= previous + 3: return "↑"
    if current <= previous - 3: return "↓"
    return "→"

# =====================================================
# HEALTHCARE DOMAIN
# =====================================================
class HealthcareDomain(BaseDomain):
    name = "healthcare"
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.shape_info = detect_dataset_shape(df)
        self.shape = self.shape_info["shape"]
        self.cols = {
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "mrn"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "condition"),
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "admission_type"),
            "volume": resolve_column(df, "total_patients") or resolve_column(df, "visits"),
            "avg_los": resolve_column(df, "avg_los"),
            "admit": resolve_column(df, "admission_date") or resolve_column(df, "date"),
            "discharge": resolve_column(df, "discharge_date"),
            "payer": resolve_column(df, "insurance") or resolve_column(df, "payer"),
            "outcome": resolve_column(df, "discharge_status"),
            "age": resolve_column(df, "age") or resolve_column(df, "dob"),
            "type": resolve_column(df, "admission_type") or resolve_column(df, "type"),
        }
        for k in ["los", "cost", "volume", "avg_los", "age"]:
            if self.cols.get(k) in df.columns: df[self.cols[k]] = pd.to_numeric(df[self.cols[k]], errors='coerce')

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
        kpis = {"dataset_shape": str(self.shape.value)}
        c = self.cols
        valid_cols = [v for v in c.values() if v]
        kpis["data_completeness"] = round(1 - df[valid_cols].isna().mean().mean(), 2) if valid_cols else 0.0

        if self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            if c["volume"]: kpis["total_patients"] = df[c["volume"]].sum()
            if c["avg_los"]: kpis["avg_los"] = df[c["avg_los"]].mean()
            kpis["is_aggregated"] = True
            return kpis

        kpis["is_aggregated"] = False
        kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)

        if c["los"] and not df[c["los"]].dropna().empty:
            kpis["avg_los"] = df[c["los"]].mean()
            kpis["long_stay_rate"] = (df[c["los"]] > 7).mean()
            kpis["benchmark_los"] = 5.0
        else: kpis["avg_los"] = None

        if c["cost"]:
            kpis["total_billing"] = df[c["cost"]].sum()
            kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
            kpis["benchmark_cost"] = df[c["cost"]].median() * 2.0 
            if kpis["avg_los"]: kpis["avg_cost_per_day"] = kpis["avg_cost_per_patient"] / kpis["avg_los"]

        if c["readmitted"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
            kpis["readmission_rate"] = df[c["readmitted"]].mean()

        if c["doctor"] and c["los"]:
            stats = df.groupby(c["doctor"])[c["los"]].mean()
            if stats.mean() > 0: kpis["provider_variance_score"] = stats.std() / stats.mean()

        if self.time_col:
            df["_dow"] = df[self.time_col].dt.dayofweek
            kpis["weekend_admission_rate"] = df["_dow"].isin([5, 6]).mean()

        # 🔥 BOARD INTELLIGENCE
        current_score = _compute_board_confidence_score(kpis)
        kpis["board_confidence_score"] = current_score
        kpis["maturity_level"] = _healthcare_maturity_level(current_score)
        kpis["board_confidence_trend"] = "→"

        return kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
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

        # 2. LOS (Active Title)
        if c["los"] and not df[c["los"]].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["los"]].hist(ax=ax, bins=15, color="teal", alpha=0.7)
                ax.axvline(VISUAL_BENCHMARK_LOS, color='red', linestyle='--', linewidth=1.5, label=f'Goal ({VISUAL_BENCHMARK_LOS}d)')
                ax.legend()
                avg = df[c["los"]].mean()
                title = f"ALERT: Avg LOS {avg:.1f} Days" if avg > 7.0 else "Length of Stay Performance"
                ax.set_title(title)
                save(fig, "los.png", "Stay duration & adherence", 0.95)
            except: pass

        # 3. Cost Drivers
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

        # 5. Efficiency Scatter
        if c["cost"] and c["los"]:
            try:
                valid = df[[c["cost"], c["los"]]].dropna()
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(valid[c["los"]], valid[c["cost"]], alpha=0.5, color="gray", s=15)
                    ax.set_title("Cost vs. LOS Correlation")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost_los.png", "Efficiency correlation", 0.85)
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
                ax.set_title("Cost Outliers Analysis")
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

        # 11. Weekend
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
        # 7 COMPOSITE INSIGHTS
        insights = []
        
        # 1. Critical Discharge Bottleneck
        if isinstance(kpis.get("long_stay_rate"), (int, float)) and kpis["long_stay_rate"] >= T["long_stay_rate_critical"]:
            insights.append({"level": "CRITICAL", "title": "Severe Discharge Bottleneck", "so_what": f"{kpis['long_stay_rate']:.1%} of patients exceed LOS targets."})
        
        # 2. Systemic Inefficiency
        if (kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"] and kpis.get("readmission_rate", 0) >= T["readmission_critical"]):
            insights.append({"level": "CRITICAL", "title": "Systemic Care Inefficiency", "so_what": "High LOS combined with high readmissions."})
        
        # 3. Efficiency Warning
        if kpis.get("avg_los") and kpis.get("benchmark_los") and kpis["avg_los"] > kpis["benchmark_los"] and not any(i["title"] == "Severe Discharge Bottleneck" for i in insights):
            insights.append({"level": "WARNING", "title": "Extended Inpatient Stay", "so_what": f"Avg LOS ({kpis['avg_los']:.1f} days) exceeds benchmarks."})
        
        # 4. Staffing Stress
        if kpis.get("weekend_admission_rate", 0) >= T["weekend_rate_warning"]:
            insights.append({"level": "WARNING", "title": "Weekend Capacity Stress", "so_what": "Weekend volume strains staffing."})
        
        # 5. Clinical Variance
        if kpis.get("provider_variance_score", 0) >= T["provider_variance_warning"]:
            insights.append({"level": "RISK", "title": "High Provider Variance", "so_what": "Significant variability in LOS across providers."})
        
        # 6. Financial Risk
        if kpis.get("avg_cost_per_patient", 0) > kpis.get("benchmark_cost", 999999) * 1.5:
            insights.append({"level": "WARNING", "title": "Cost Anomaly", "so_what": "Costs significantly exceed benchmarks."})

        # 7. Data Risk
        if kpis.get("data_completeness", 1) < 0.90:
            insights.append({"level": "RISK", "title": "Data Integrity Gap", "so_what": "Missing clinical fields limit precision."})

        if not insights:
            insights.append({"level": "INFO", "title": "Stable Operations", "so_what": "Metrics within tolerance."})
        return insights

    def generate_recommendations(self, df, kpis, insights=None, shape_info=None):
        # 7 RECOMMENDATION RULES (Return Top 5)
        recs = []
        titles = [i["title"] for i in (insights or [])]

        if "Severe Discharge Bottleneck" in titles:
            recs.append({"action": "Audit discharge planning", "priority": "HIGH", "timeline": "30 days", "owner": "Clinical Ops", "expected_outcome": "Reduce LOS < 6 days"})
        
        if kpis.get("readmission_rate", 0) > T["readmission_warning"]:
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
    TOKENS = {"patient", "admission", "diagnosis", "medical_condition", "clinical", "doctor", "insurance"}
    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        return DomainDetectionResult("healthcare", min(len(hits)/3, 1.0), {})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
