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
    row_count = len(df)

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
    if score[best_shape] == 0: best_shape = DatasetShape.UNKNOWN

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
# HEALTHCARE DOMAIN (FINAL 10/10)
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Universal Healthcare Intelligence (Adaptive)"

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        shape_result = detect_dataset_shape(df)
        self.shape = shape_result["shape"]
        
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
        }

        if self.shape == DatasetShape.ROW_LEVEL_CLINICAL:
            if not self.cols["los"] and self.cols["admit"] and self.cols["discharge"]:
                df["derived_los"] = _derive_length_of_stay(df, self.cols["admit"], self.cols["discharge"])
                self.cols["los"] = "derived_los"
            
            if self.cols["readmitted"]:
                val_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}
                if df[self.cols["readmitted"]].dtype == object:
                    df[self.cols["readmitted"]] = df[self.cols["readmitted"]].astype(str).str.lower().map(val_map).fillna(0)

        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols
        
        # Metadata
        kpis["dataset_shape"] = self.shape.value

        # --- Aggregated Path ---
        if self.shape == DatasetShape.AGGREGATED_OPERATIONAL:
            if c["volume"]: kpis["total_patients"] = df[c["volume"]].sum()
            if c["avg_los"]: kpis["avg_los"] = df[c["avg_los"]].mean()
            kpis["is_aggregated"] = True
            return kpis

        # --- Row-Level Path ---
        kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)
        kpis["is_aggregated"] = False

        if c["los"] and not df[c["los"]].dropna().empty:
            kpis["avg_los"] = df[c["los"]].mean()
            median_los = df[c["los"]].median()
            kpis["benchmark_los"] = median_los * 1.5 if len(df) > 10 else 7.0
            kpis["long_stay_rate"] = (df[c["los"]] > kpis["benchmark_los"]).mean()
            
            if kpis["avg_los"] > 0:
                kpis["bed_turnover_index"] = _safe_div(1, kpis["avg_los"])
        else:
            kpis["avg_los"] = None

        if c["readmitted"]:
            kpis["readmission_rate"] = df[c["readmitted"]].mean()

        if c["cost"]:
            kpis["total_billing"] = df[c["cost"]].sum()
            kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
            kpis["benchmark_cost"] = df[c["cost"]].median() * 2.0 if len(df) > 10 else 50000
            
            if kpis.get("avg_los"):
                kpis["avg_cost_per_day"] = _safe_div(kpis["avg_cost_per_patient"], kpis["avg_los"])

        # Provider Variance (FIX 5: Volume Threshold)
        if c["doctor"] and c["los"]:
            provider_counts = df[c["doctor"]].value_counts()
            eligible = provider_counts[provider_counts >= 5].index # Min 5 patients
            if not eligible.empty:
                stats = df[df[c["doctor"]].isin(eligible)].groupby(c["doctor"])[c["los"]].mean()
                if len(stats) > 1:
                    kpis["provider_los_std"] = stats.std()
                    kpis["max_provider_los"] = stats.max()
                    kpis["median_provider_los"] = stats.median()

        # 🛑 FIX 1: Sanitize and RETURN KPIS (Not 'k')
        for k_key, v in list(kpis.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                kpis[k_key] = None

        return kpis  # Correct return variable

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
            fig, ax = plt.subplots(figsize=(7, 4))
            col = c["volume"] if c["volume"] else c["pid"]
            agg = "sum" if c["volume"] else "count"
            # Future-proof 'ME'
            if agg == "sum":
                df.set_index(pd.to_datetime(df[self.time_col])).resample('ME')[col].sum().plot(ax=ax, color="#1f77b4")
            else:
                df.set_index(pd.to_datetime(df[self.time_col])).resample('ME').size().plot(ax=ax, color="#1f77b4")
            ax.set_title("Patient Volume Trend")
            save(fig, "vol_trend.png", "Demand over time", 0.9)

        # 2. LOS Distribution
        if is_row_level and c["los"] and not df[c["los"]].dropna().empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = min(15, max(5, int(len(df) ** 0.5)))
            df[c["los"]].dropna().hist(ax=ax, bins=bins, color="teal")
            ax.set_title("Length of Stay (Days)")
            save(fig, "los_dist.png", "Stay duration", 0.85)

        # 3. Cost by Condition (FIX 4: Aggregation Safe)
        if c["diagnosis"] and c["cost"] and not kpis.get("is_aggregated"):
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Avg Cost by Condition")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "cost_diag.png", "Cost drivers", 0.9)

        # 4. Readmission Rate
        if c["readmitted"] and c["diagnosis"] and is_row_level:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax, color="red")
            ax.set_title("Highest Readmission Conditions")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
            save(fig, "readm_diag.png", "Clinical risk", 0.95)

        # 5. Cost vs LOS
        if is_row_level and c["cost"] and c["los"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_df = df.sample(1000) if len(df) > 1000 else df
            ax.scatter(plot_df[c["los"]], plot_df[c["cost"]], alpha=0.5, color="gray")
            ax.set_title("Cost vs LOS")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "cost_los.png", "Efficiency", 0.8)

        # 6. Provider Performance
        if c["doctor"] and c["los"] and is_row_level:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot(kind="bar", ax=ax, color="brown")
            ax.set_title("Avg LOS by Provider")
            save(fig, "provider_los.png", "Care consistency", 0.7)

        # 7. Payer Mix
        if c["payer"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
            ax.axis('equal')
            ax.set_title("Payer Mix")
            save(fig, "payer_mix.png", "Revenue source", 0.65)

        # 8. Outcomes
        if c["outcome"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["outcome"]].value_counts().head(5).plot(kind="bar", ax=ax, color="purple")
            ax.set_title("Discharge Outcomes")
            save(fig, "outcomes.png", "Disposition", 0.75)

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:6] # FIX 3: Increased Visual Cap

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        c = self.cols
        
        los = kpis.get("avg_los")
        readm = kpis.get("readmission_rate", 0)
        cost = kpis.get("avg_cost_per_patient", 0)
        long_rate = kpis.get("long_stay_rate", 0)
        
        # 1. Dataset Context
        if self.shape != DatasetShape.ROW_LEVEL_CLINICAL:
            insights.append({"level": "INFO", "title": f"Mode: {self.shape.name}", "so_what": "Analysis adapted for dataset shape."})

        # 2. Missing Data
        if los is None and not kpis.get("is_aggregated"):
            insights.append({"level": "RISK", "title": "Missing LOS Data", "so_what": "Operational efficiency cannot be assessed."})

        # Dynamic Thresholds
        limit_los = kpis.get("benchmark_los", 7.0)
        limit_cost = kpis.get("benchmark_cost", 50000)

        # Composite: Strain
        if los and los > limit_los and readm > 0.15:
            insights.append({"level": "CRITICAL", "title": "Operational Strain", "so_what": "High LOS and Readmissions indicate systemic congestion."})

        # Composite: Inefficiency
        if cost > limit_cost and readm > 0.15:
            insights.append({"level": "WARNING", "title": "Financial Inefficiency", "so_what": "High costs yielding poor stability."})

        # Atomic Fallbacks (FIX 2: Smart Suppression)
        existing_titles = [i["title"] for i in insights]

        if readm > 0.15 and "Operational Strain" not in existing_titles and "Financial Inefficiency" not in existing_titles:
            insights.append({"level": "RISK", "title": "High Readmission Rate", "so_what": f"Readmission rate is {readm:.1%}."})

        if los and los > limit_los and "Operational Strain" not in existing_titles:
            insights.append({"level": "WARNING", "title": "Extended Length of Stay", "so_what": f"Avg stay is {los:.1f} days."})

        # Problem Child
        if c["diagnosis"] and c["cost"] and c["readmitted"]:
            count_col = c["pid"] if c["pid"] else c["diagnosis"]
            stats = df.groupby(c["diagnosis"]).agg({c["cost"]: "mean", c["readmitted"]: "mean", count_col: "count"})
            significant = stats[stats[count_col] > len(df) * 0.05]
            
            if not significant.empty:
                significant["impact"] = significant[c["cost"]] * significant[c["readmitted"]]
                top = significant["impact"].idxmax()
                if significant.loc[top, c["cost"]] > 8000 and significant.loc[top, c["readmitted"]] > 0.10:
                    insights.append({"level": "WARNING", "title": f"High-Impact: {top}", "so_what": f"{top} drives cost and risk."})

        if not insights:
            insights.append({"level": "INFO", "title": "Stable", "so_what": "Metrics within tolerance."})

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df, kpis, insights=None):
        recs = []
        if insights is None: insights = self.generate_insights(df, kpis)
        titles = [i["title"] for i in insights]

        if "Operational Strain" in titles: recs.append({"action": "Audit discharge planning.", "priority": "HIGH"})
        if "Missing LOS Data" in titles: recs.append({"action": "Verify data ingestion.", "priority": "HIGH"})
        if kpis.get("readmission_rate", 0) > 0.15: recs.append({"action": "Implement post-discharge calls.", "priority": "MEDIUM"})

        return recs or [{"action": "Monitor trends.", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
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
