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

# =====================================================
# SHAPE DETECTION (UNIVERSAL)
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

    # Extended fuzzy logic for universal columns
    if any(x in c for c in cols for x in ["patient", "mrn", "pid", "id", "name"]): 
        score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
    if any(x in c for c in cols for x in ["admit", "admission", "date", "joining"]): 
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
# HEALTHCARE DOMAIN (FACT ENGINE)
# =====================================================
class HealthcareDomain(BaseDomain):
    name = "healthcare"
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.shape_info = detect_dataset_shape(df)
        self.shape = self.shape_info["shape"]
        
        # UNIVERSAL COLUMN MAPPING (Handles "Medical Condition", "Billing Amount" etc.)
        self.cols = {
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "name") or resolve_column(df, "mrn"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition") or resolve_column(df, "condition"),
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost") or resolve_column(df, "charges"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "admission_type") or resolve_column(df, "readmission"),
            "volume": resolve_column(df, "total_patients") or resolve_column(df, "visits"),
            "avg_los": resolve_column(df, "avg_los"),
            "admit": resolve_column(df, "date_of_admission") or resolve_column(df, "admission_date") or resolve_column(df, "joining_date"),
            "discharge": resolve_column(df, "discharge_date") or resolve_column(df, "date_of_discharge"),
            "payer": resolve_column(df, "insurance_provider") or resolve_column(df, "payer") or resolve_column(df, "insurance"),
            "outcome": resolve_column(df, "discharge_status") or resolve_column(df, "outcome") or resolve_column(df, "test_results"),
            "age": resolve_column(df, "age") or resolve_column(df, "dob"),
            "gender": resolve_column(df, "gender") or resolve_column(df, "sex"),
            "type": resolve_column(df, "admission_type") or resolve_column(df, "type"),
        }
        
        # Numeric Force
        for k in ["los", "cost", "volume", "avg_los", "age"]:
            if self.cols.get(k) in df.columns:
                df[self.cols[k]] = pd.to_numeric(df[self.cols[k]], errors='coerce')

        # Logic: Calculate LOS if missing (Requires Admit + Discharge)
        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL and not self.cols["los"] and self.cols["admit"] and self.cols["discharge"]:
            try:
                a = pd.to_datetime(df[self.cols["admit"]], errors="coerce")
                d = pd.to_datetime(df[self.cols["discharge"]], errors="coerce")
                df["derived_los"] = (d - a).dt.days
                self.cols["los"] = "derived_los"
            except: pass

        # Logic: Time sorting
        if self.cols["admit"]:
            df[self.cols["admit"]] = pd.to_datetime(df[self.cols["admit"]], errors="coerce")
            self.time_col = self.cols["admit"]
            df = df.sort_values(self.time_col)
        else:
            self.time_col = None

        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        raw_kpis = {"dataset_shape": str(self.shape.value)}
        c = self.cols
        
        # 1. Completeness Score
        valid_cols = [v for v in c.values() if v]
        raw_kpis["data_completeness"] = round(1 - df[valid_cols].isna().mean().mean(), 2) if valid_cols else 0.0

        # 2. Aggregated Mode
        if self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            if c["volume"]: raw_kpis["total_patients"] = df[c["volume"]].sum()
            if c["avg_los"]: raw_kpis["avg_los"] = df[c["avg_los"]].mean()
            raw_kpis["is_aggregated"] = True
            return raw_kpis

        # 3. Row Level Mode
        raw_kpis["is_aggregated"] = False
        raw_kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)

        if c["los"] and not df[c["los"]].dropna().empty:
            raw_kpis["avg_los"] = df[c["los"]].mean()
            raw_kpis["long_stay_rate"] = (df[c["los"]] > 7).mean()
        else:
            raw_kpis["avg_los"] = None

        if c["cost"]:
            raw_kpis["total_billing"] = df[c["cost"]].sum()
            raw_kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
            raw_kpis["benchmark_cost"] = df[c["cost"]].median() * 2.0 

        if c["readmitted"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
            raw_kpis["readmission_rate"] = df[c["readmitted"]].mean()

        if c["age"]:
            raw_kpis["avg_patient_age"] = df[c["age"]].mean()

        if c["doctor"] and c["los"]:
            stats = df.groupby(c["doctor"])[c["los"]].mean()
            if stats.mean() > 0:
                raw_kpis["provider_variance_score"] = stats.std() / stats.mean()

        # ORDERING: This ensures the "Most Important" 3-5 appear first in the report
        ordered_keys = ["total_patients", "avg_cost_per_patient", "avg_los", "readmission_rate", "long_stay_rate", "data_completeness", "provider_variance_score", "avg_patient_age", "total_billing"]
        final_kpis = {k: raw_kpis[k] for k in ordered_keys if k in raw_kpis and not (isinstance(raw_kpis[k], float) and (np.isnan(raw_kpis[k]) or np.isinf(raw_kpis[k])))}
        
        # Add any remaining keys
        for k, v in raw_kpis.items():
            if k not in final_kpis and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                final_kpis[k] = v
                
        return final_kpis

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

        # --- 11 POTENTIAL VISUALS (Filtered to Top 6 later) ---

        # 1. Volume Trend
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                df.set_index(self.time_col).resample('ME').size().plot(ax=ax, color="#1f77b4")
                ax.set_title("Patient Volume Trend")
                save(fig, "vol.png", "Demand stability", 0.99)
            except: pass

        # 2. LOS Distribution
        if c["los"] and not df[c["los"]].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["los"]].hist(ax=ax, bins=15, color="teal")
                ax.set_title("LOS Distribution")
                save(fig, "los.png", "Stay duration", 0.95)
            except: pass

        # 3. Cost by Condition
        if c["diagnosis"] and c["cost"]:
            try:
                if not df[c["cost"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot(kind="bar", ax=ax, color="orange")
                    ax.set_title("Cost by Condition")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost.png", "Cost drivers", 0.90)
            except: pass

        # 4. Readmission by Condition
        if c["readmitted"] and c["diagnosis"] and pd.api.types.is_numeric_dtype(df[c["readmitted"]]):
            try:
                if not df[c["readmitted"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax, color="red")
                    ax.set_title("Readmission Risk by Condition")
                    save(fig, "readm.png", "Clinical risk", 0.88)
            except: pass

        # 5. Cost vs LOS
        if c["cost"] and c["los"]:
            try:
                valid = df[[c["cost"], c["los"]]].dropna()
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(valid[c["los"]], valid[c["cost"]], alpha=0.5, color="gray", s=15)
                    ax.set_title("Cost vs. LOS")
                    ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost_los.png", "Efficiency", 0.85)
            except: pass

        # 6. Provider Performance
        if c["doctor"] and c["los"]:
            try:
                if not df[c["los"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot(kind="bar", ax=ax, color="brown")
                    ax.set_title("Avg LOS by Provider")
                    save(fig, "prov.png", "Care consistency", 0.80)
            except: pass

        # 7. Payer Mix
        if c["payer"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
                ax.axis('equal')
                ax.set_title("Payer Mix")
                save(fig, "payer.png", "Revenue source", 0.75)
            except: pass

        # 8. Outcomes
        if c["outcome"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["outcome"]].value_counts().head(5).plot(kind="bar", ax=ax, color="purple")
                ax.set_title("Discharge Outcomes")
                save(fig, "out.png", "Disposition", 0.70)
            except: pass

        # 9. Cost Boxplot
        if c["cost"]:
            try:
                if not df[c["cost"]].dropna().empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.boxplot(df[c["cost"]].dropna(), vert=False)
                    ax.set_title("Cost Distribution")
                    ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                    save(fig, "cost_box.png", "Financial outliers", 0.65)
            except: pass

        # 10. Age Distribution (New)
        if c["age"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["age"]].hist(ax=ax, bins=20, color="green", alpha=0.6)
                ax.set_title("Patient Age Distribution")
                save(fig, "age.png", "Demographics", 0.60)
            except: pass

        # 11. Admission Type (New)
        if c["type"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["type"]].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%')
                ax.set_title("Admission Type")
                save(fig, "type.png", "Acuity mix", 0.55)
            except: pass

        if len(visuals) < 3:
            visuals.append({"path": "placeholder.png", "caption": "Limited data for visuals", "importance": 0.1})
            
        # Return Top 6 as requested
        return sorted(visuals, key=lambda x: x["importance"], reverse=True)[:6]

    def generate_insights(self, df, kpis, shape_info=None):
        # FACT-BASED ONLY. Narrative Engine adds Judgment.
        insights = []
        c = self.cols
        
        # Need 7 potential insights for robust reporting
        
        # 1. Limited Visibility (Shape)
        if self.shape == DatasetShape.UNKNOWN:
            insights.append({"level": "WARNING", "title": "Limited Visibility", "so_what": "Dataset lacks granular clinical fields."})

        # 2. Payer Concentration Risk
        if c["payer"]:
            top_payer = df[c["payer"]].value_counts(normalize=True).iloc[0]
            if top_payer > 0.5:
                insights.append({"level": "RISK", "title": "Payer Concentration", "so_what": f"Top payer holds {top_payer:.0%} of volume, creating revenue risk."})

        # 3. Geriatric Complexity (Age + LOS)
        if c["age"] and c["los"]:
            avg_age = df[c["age"]].mean()
            if avg_age > 65 and kpis.get("avg_los", 0) > 6:
                insights.append({"level": "WARNING", "title": "Geriatric Complexity", "so_what": "High average age (65+) correlates with extended LOS."})

        # 4. Weekend Surge
        if kpis.get("weekend_admission_rate", 0) > 0.35:
             insights.append({"level": "WARNING", "title": "Weekend Surge", "so_what": "Weekend admissions exceed 35%, straining skeleton staff."})

        # 5. Clinical Variation
        if kpis.get("provider_variance_score", 0) > 0.5:
            insights.append({"level": "RISK", "title": "Clinical Variation", "so_what": "High inconsistency in LOS across providers."})

        # 6. Readmission Spike
        if kpis.get("readmission_rate", 0) > 0.15:
            insights.append({"level": "CRITICAL", "title": "High Readmission Rate", "so_what": "Readmission rate > 15% indicates discharge gaps."})

        # 7. Cost Anomaly
        if kpis.get("avg_cost_per_patient", 0) > 20000:
             insights.append({"level": "WARNING", "title": "High Cost Base", "so_what": "Average cost per patient exceeds $20k."})

        return insights

    def generate_recommendations(self, df, kpis, insights=None, shape_info=None):
        recs = []
        titles = [i["title"] for i in (insights or [])]

        # 7 Potential Recommendations
        
        # 1. Data Integrity
        if kpis.get("data_completeness", 1) < 0.9:
            recs.append({"action": "Verify ETL timestamps", "priority": "HIGH", "timeline": "Immediate", "owner": "Data Eng", "expected_outcome": "Enable Analysis"})
        
        # 2. Revenue Recovery
        if kpis.get("avg_cost_per_patient", 0) > 50000:
            recs.append({"action": "Review high-cost outliers", "priority": "MEDIUM", "timeline": "30 days", "owner": "Finance", "expected_outcome": "Recover Revenue"})
        
        # 3. Discharge Planning
        if kpis.get("long_stay_rate", 0) > 0.15:
            recs.append({"action": "Audit discharge planning", "priority": "HIGH", "timeline": "30 days", "owner": "Clinical Ops", "expected_outcome": "Reduce LOS"})

        # 4. Payer Strategy
        if self.cols["payer"]:
            recs.append({"action": "Evaluate payer contracts", "priority": "LOW", "timeline": "Annual", "owner": "Revenue Cycle", "expected_outcome": "Optimize Profit"})

        # 5. Geriatric Care
        if "Geriatric Complexity" in titles:
            recs.append({"action": "Implement geriatric pathways", "priority": "MEDIUM", "timeline": "90 days", "owner": "CMO", "expected_outcome": "Reduce LoS for Seniors"})

        # 6. Staffing
        if "Weekend Surge" in titles:
            recs.append({"action": "Adjust weekend staffing", "priority": "MEDIUM", "timeline": "Next Quarter", "owner": "HR", "expected_outcome": "Capacity Balance"})

        # 7. Standardization
        if "Clinical Variation" in titles:
            recs.append({"action": "Standardize treatment protocols", "priority": "MEDIUM", "timeline": "60 days", "owner": "CMO", "expected_outcome": "Consistent Care"})

        return recs or [{"action": "Monitor trends", "priority": "LOW", "timeline": "Ongoing", "owner": "Ops", "expected_outcome": "Stability"}]

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "medical_condition", "clinical", "doctor", "insurance"}
    
    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        return DomainDetectionResult("healthcare", min(len(hits)/3, 1.0), {})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
