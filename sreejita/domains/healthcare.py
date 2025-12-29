import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from enum import Enum
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# SHAPE DETECTION ENGINE
# =====================================================

class DatasetShape(str, Enum):
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    QUALITY_METRICS = "quality_metrics"
    UNKNOWN = "unknown"

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    cols = set(c.lower() for c in df.columns)
    
    score = {
        DatasetShape.ROW_LEVEL_CLINICAL: 0,
        DatasetShape.AGGREGATED_OPERATIONAL: 0,
        DatasetShape.FINANCIAL_SUMMARY: 0,
        DatasetShape.QUALITY_METRICS: 0,
    }

    if any(c in cols for c in ["patient_id", "mrn", "pid"]): score[DatasetShape.ROW_LEVEL_CLINICAL] += 3
    if any(c in cols for c in ["admission_date", "discharge_date"]): score[DatasetShape.ROW_LEVEL_CLINICAL] += 2
    
    if any(c in cols for c in ["total_patients", "visits", "volume"]): score[DatasetShape.AGGREGATED_OPERATIONAL] += 3
    if any(c in cols for c in ["revenue", "billing", "total_cost"]): score[DatasetShape.FINANCIAL_SUMMARY] += 2
    if any("rate" in c for c in cols): score[DatasetShape.QUALITY_METRICS] += 2

    best_shape = max(score, key=score.get)

    # Priority Override (Row-level beats everything)
    if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 3:
        best_shape = DatasetShape.ROW_LEVEL_CLINICAL

    if score[best_shape] == 0: 
        best_shape = DatasetShape.UNKNOWN

    return {"shape": best_shape, "score": score}


# =====================================================
# HELPERS
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d

def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["admission_date", "visit_date", "date", "discharge_date"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="coerce")
                return c
            except:
                continue
    return None

def _derive_length_of_stay(df, admit, discharge):
    if admit and discharge:
        try:
            a = pd.to_datetime(df[admit], errors="coerce")
            d = pd.to_datetime(df[discharge], errors="coerce")
            los = (d - a).dt.days
            return los.where(los >= 0)
        except:
            pass
    return None


# =====================================================
# HEALTHCARE DOMAIN (FINAL POLISHED)
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Universal Healthcare Intelligence (Adaptive)"

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        # ✅ Store full result for debugging
        self.shape_info = detect_dataset_shape(df)
        self.shape = self.shape_info["shape"]
        
        self.cols = {
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "mrn"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "condition"),
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "readmission"),
            "volume": resolve_column(df, "total_patients") or resolve_column(df, "visits"),
            "avg_los": resolve_column(df, "avg_los") or resolve_column(df, "average_los"),
            "admit": resolve_column(df, "admission_date"),
            "discharge": resolve_column(df, "discharge_date"),
            "payer": resolve_column(df, "insurance") or resolve_column(df, "payer"),
            "outcome": resolve_column(df, "discharge_status") or resolve_column(df, "outcome"),
            "age": resolve_column(df, "age") or resolve_column(df, "dob"),
        }

        # Strict Numeric Enforcement
        numeric_targets = ["los", "cost", "volume", "avg_los", "age"]
        for key in numeric_targets:
            col_name = self.cols.get(key)
            if col_name and col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        # Readmission Coercion Safety
        if self.cols["readmitted"]:
            col_r = self.cols["readmitted"]
            val_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}
            if df[col_r].dtype == object:
                df[col_r] = df[col_r].astype(str).str.lower().map(val_map)
            df[col_r] = pd.to_numeric(df[col_r], errors='coerce')

        # Row Level Logic
        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL:
            if not self.cols["los"] and self.cols["admit"] and self.cols["discharge"]:
                df["derived_los"] = _derive_length_of_stay(df, self.cols["admit"], self.cols["discharge"])
                self.cols["los"] = "derived_los"

        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols
        kpis["dataset_shape"] = self.shape.value
        
        # ✅ OPTIONAL 1: Expose scoring debug info
        kpis["debug_shape_score"] = self.shape_info.get("score")

        if self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            if c["volume"]: kpis["total_patients"] = df[c["volume"]].sum()
            if c["avg_los"]: kpis["avg_los"] = df[c["avg_los"]].mean()
            kpis["is_aggregated"] = True
            
        else:
            # --- Row-Level Logic ---
            kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)
            kpis["is_aggregated"] = False

            # LOS Metrics
            if c["los"] and not df[c["los"]].dropna().empty:
                kpis["avg_los"] = df[c["los"]].mean()
                median_los = df[c["los"]].median()
                kpis["benchmark_los"] = median_los * 1.5 if len(df) > 10 else 7.0
                kpis["long_stay_rate"] = (df[c["los"]] > kpis["benchmark_los"]).mean()
                if kpis["avg_los"] > 0:
                    kpis["bed_turnover_index"] = _safe_div(1, kpis["avg_los"])
            else:
                kpis["avg_los"] = None

            # Financial Metrics
            if c["cost"]:
                kpis["total_billing"] = df[c["cost"]].sum()
                kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
                kpis["max_single_cost"] = df[c["cost"]].max()
                kpis["benchmark_cost"] = df[c["cost"]].median() * 2.0 if len(df) > 10 else 50000
                if kpis.get("avg_los"):
                    kpis["avg_cost_per_day"] = _safe_div(kpis["avg_cost_per_patient"], kpis["avg_los"])

            # Quality Metrics
            if c["readmitted"]:
                kpis["readmission_rate"] = df[c["readmitted"]].mean()
            
            if c["outcome"]:
                neg_mask = df[c["outcome"]].astype(str).str.lower().str.contains(r'died|expired|death|mortality')
                kpis["mortality_rate"] = neg_mask.mean()

            # Weekend Rate
            if self.time_col and self.shape == DatasetShape.ROW_LEVEL_CLINICAL:
                df["_dow"] = df[self.time_col].dt.dayofweek
                kpis["weekend_admission_rate"] = df["_dow"].isin([5, 6]).mean()

            # Provider Variance
            if c["doctor"] and c["los"]:
                provider_counts = df[c["doctor"]].value_counts()
                eligible = provider_counts[provider_counts >= 5].index 
                if not eligible.empty:
                    stats = df[df[c["doctor"]].isin(eligible)].groupby(c["doctor"])[c["los"]].mean()
                    mean_val = stats.mean()
                    if mean_val and mean_val > 0:
                        kpis["provider_los_std"] = stats.std()
                        kpis["provider_variance_score"] = stats.std() / mean_val
                    else:
                        kpis["provider_variance_score"] = None

        # Sanitize NaNs
        for k_key, v in list(kpis.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                kpis[k_key] = None

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        kpis = self.calculate_kpis(df)

        def save(fig, name, caption, imp):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({"path": str(p), "caption": caption, "importance": imp})

        def human_fmt(x, _):
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        is_row_level = self.shape == DatasetShape.ROW_LEVEL_CLINICAL
        
        # 1. Volume Trend
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                col = c["volume"] if c["volume"] else c["pid"]
                agg = "sum" if c["volume"] else "count"
                if agg == "sum":
                    df.set_index(pd.to_datetime(df[self.time_col])).resample('ME')[col].sum().plot(ax=ax, color="#1f77b4")
                else:
                    df.set_index(pd.to_datetime(df[self.time_col])).resample('ME').size().plot(ax=ax, color="#1f77b4")
                ax.set_title("Patient Volume Trend")
                save(fig, "vol_trend.png", "Demand over time", 0.95)
            except: pass

        # 2. LOS Distribution
        if is_row_level and c["los"] and not df[c["los"]].dropna().empty:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                bins = min(15, max(5, int(len(df) ** 0.5)))
                df[c["los"]].dropna().hist(ax=ax, bins=bins, color="teal", alpha=0.7)
                ax.set_title("Length of Stay Distribution")
                ax.set_xlabel("Days")
                save(fig, "los_dist.png", "Stay duration spread", 0.90)
            except: pass

        # 3. Cost by Condition
        if c["diagnosis"] and c["cost"] and not kpis.get("is_aggregated"):
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot(kind="bar", ax=ax, color="orange")
                ax.set_title("Avg Cost by Condition")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "cost_diag.png", "Top cost drivers", 0.88)
            except: pass

        # 4. Readmission by Condition
        if c["readmitted"] and c["diagnosis"] and is_row_level:
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax, color="red")
                ax.set_title("Conditions with Highest Readmission")
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
                save(fig, "readm_diag.png", "Clinical risk areas", 0.85)
            except: pass

        # 5. Cost vs LOS Scatter
        if is_row_level and c["cost"] and c["los"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                plot_df = df.sample(1000) if len(df) > 1000 else df
                ax.scatter(plot_df[c["los"]], plot_df[c["cost"]], alpha=0.5, color="gray", s=15)
                ax.set_title("Cost vs. Length of Stay")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "cost_los.png", "Efficiency correlation", 0.80)
            except: pass

        # 6. Provider Performance
        if c["doctor"] and c["los"] and is_row_level:
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot(kind="bar", ax=ax, color="brown")
                ax.set_title("Avg LOS by Provider")
                save(fig, "provider_los.png", "Care consistency", 0.75)
            except: pass

        # 7. Payer Mix
        if c["payer"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
                ax.axis('equal')
                ax.set_title("Payer Mix")
                save(fig, "payer_mix.png", "Revenue source", 0.70)
            except: pass

        # 8. Outcomes
        if c["outcome"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["outcome"]].value_counts().head(5).plot(kind="bar", ax=ax, color="purple")
                ax.set_title("Discharge Outcomes")
                save(fig, "outcomes.png", "Patient disposition", 0.65)
            except: pass

        # 9. Day of Week Pattern
        if self.time_col:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                dow_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
                df["_dow"] = df[self.time_col].dt.dayofweek
                counts = df["_dow"].value_counts().sort_index()
                counts.index = counts.index.map(dow_map)
                counts.plot(kind='bar', ax=ax, color="#2ca02c")
                ax.set_title("Admissions by Day of Week")
                save(fig, "admit_dow.png", "Weekly volume pattern", 0.60)
            except: pass
        
        # 10. Cost Boxplot
        if c["cost"] and is_row_level:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.boxplot(df[c["cost"]].dropna(), vert=False)
                ax.set_title("Cost Distribution (Outliers)")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "cost_box.png", "Financial outliers", 0.55)
            except: pass

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:6]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any], shape_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        insights = []
        c = self.cols
        
        los = kpis.get("avg_los")
        readm = kpis.get("readmission_rate", 0)
        cost = kpis.get("avg_cost_per_patient", 0)
        weekend_rate = kpis.get("weekend_admission_rate", 0)
        prov_var = kpis.get("provider_variance_score", 0)
        
        limit_los = kpis.get("benchmark_los", 7.0)
        limit_cost = kpis.get("benchmark_cost", 50000)

        # 1. Dataset Context
        if self.shape != DatasetShape.ROW_LEVEL_CLINICAL:
            insights.append({"level": "INFO", "title": f"Mode: {self.shape.name}", "so_what": "Analysis adapted for dataset shape."})

        # 2. Composite: "Triple Threat"
        if (los and los > limit_los) and (cost > limit_cost) and (readm > 0.15):
            insights.append({
                "level": "CRITICAL", 
                "title": "Systemic Performance Drag", 
                "so_what": "Convergence of high cost, long stays, and readmissions indicates deep operational inefficiency."
            })
        
        # 3. Composite: "Operational Strain"
        elif (los and los > limit_los) and (readm > 0.12):
            insights.append({
                "level": "CRITICAL", 
                "title": "Operational Strain", 
                "so_what": "High LOS combined with readmissions suggests premature discharge or poor recovery planning."
            })

        # 4. Composite: "Clinical Variation"
        if prov_var and prov_var > 0.5:
            insights.append({
                "level": "RISK", 
                "title": "High Clinical Variation", 
                "so_what": "Significant inconsistency in LOS across providers suggests lack of standardized protocols."
            })

        # 5. Composite: "Weekend Effect"
        if weekend_rate > 0.35:
            insights.append({
                "level": "WARNING", 
                "title": "Weekend Surge Detected", 
                "so_what": f"Weekend admissions ({weekend_rate:.1%}) exceed norms, potentially straining skeleton staff."
            })

        # 6. Composite: "Financial Inefficiency"
        if cost and cost > limit_cost * 1.2:
            insights.append({
                "level": "WARNING", 
                "title": "Cost Anomaly", 
                "so_what": f"Avg cost per patient (${cost:,.0f}) is significantly above median benchmarks."
            })

        # 7. Atomic Fallbacks
        if readm and readm > 0.15 and not any(i["title"] == "Systemic Performance Drag" for i in insights):
            insights.append({"level": "RISK", "title": "High Readmission Rate", "so_what": f"Readmission rate is {readm:.1%}."})

        if not insights:
            insights.append({"level": "INFO", "title": "Stable Operations", "so_what": "Key metrics are within expected tolerance levels."})

        # Deduplication
        seen = set()
        final_insights = []
        for i in insights:
            if i["title"] not in seen:
                final_insights.append(i)
                seen.add(i["title"])
        
        return final_insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df, kpis, insights=None, shape_info=None):
        if insights is None: insights = self.generate_insights(df, kpis, shape_info)
        titles = [i["title"] for i in insights]
        recs = []

        # 1. Strain / Readmissions
        if "Operational Strain" in titles or "Systemic Performance Drag" in titles:
            recs.append({
                "action": "Audit discharge planning process immediately.",
                "priority": "HIGH",
                "timeline": "30 days"
            })
            recs.append({
                "action": "Implement mandatory post-discharge follow-up calls.",
                "priority": "HIGH",
                "timeline": "Immediate"
            })

        # 2. Clinical Variation
        if "High Clinical Variation" in titles:
            recs.append({
                "action": "Standardize treatment protocols for top conditions.",
                "priority": "MEDIUM",
                "timeline": "90 days"
            })
            recs.append({
                "action": "Conduct peer review for outlier providers.",
                "priority": "MEDIUM",
                "timeline": "60 days"
            })

        # 3. Weekend Surge
        if "Weekend Surge Detected" in titles:
            recs.append({
                "action": "Adjust staffing rosters for weekend coverage.",
                "priority": "MEDIUM",
                "timeline": "Next Quarter"
            })

        # 4. Financial
        if kpis.get("avg_cost_per_patient", 0) > 50000:
            recs.append({
                "action": "Review high-cost outliers for billing errors.",
                "priority": "MEDIUM",
                "timeline": "30 days"
            })

        # 5. Data Hygiene
        if kpis.get("avg_los") is None:
            recs.append({
                "action": "Verify admission/discharge timestamps in ETL.",
                "priority": "HIGH",
                "timeline": "Immediate"
            })

        # 6. General Efficiency
        if kpis.get("bed_turnover_index", 0) < 0.1 and kpis.get("bed_turnover_index") is not None:
            recs.append({
                "action": "Optimize bed management and turnover workflow.",
                "priority": "LOW",
                "timeline": "6 months"
            })

        # 7. Payer/Revenue (✅ OPTIONAL 2: Guard against small datasets)
        if self.cols["payer"] and kpis.get("total_patients", 0) > 50:
            recs.append({
                "action": "Evaluate payer contracts against readmission risks.",
                "priority": "LOW",
                "timeline": "Annual"
            })

        if not recs:
            recs.append({"action": "Monitor operational trends.", "priority": "LOW", "timeline": "Ongoing"})

        return recs


# =====================================================
# DOMAIN REGISTRATION
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "clinical", "doctor", "readmitted"}

    def detect(self, df) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        return DomainDetectionResult("healthcare", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
