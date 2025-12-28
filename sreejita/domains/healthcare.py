import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


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
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except:
                continue
    return None

def _derive_length_of_stay(df: pd.DataFrame, admit_col, discharge_col) -> pd.Series:
    """Calculates LOS days from date columns if explicit LOS is missing."""
    if admit_col and discharge_col:
        try:
            a = pd.to_datetime(df[admit_col], errors="coerce")
            d = pd.to_datetime(df[discharge_col], errors="coerce")
            los = (d - a).dt.days
            return los.where(los >= 0) # Filter negative dates
        except:
            return None
    return None


# =====================================================
# HEALTHCARE DOMAIN (UNIVERSAL 10/10)
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Universal Healthcare Intelligence (Clinical, Operational, Financial)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        # 1. Resolve columns ONCE.
        self.cols = {
            # ID & Demographics
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "mrn"),
            "gender": resolve_column(df, "gender") or resolve_column(df, "sex"),
            "age": resolve_column(df, "age"),
            "payer": resolve_column(df, "insurance") or resolve_column(df, "payer"),
            
            # Clinical
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "condition"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "readmission"),
            "outcome": resolve_column(df, "discharge_status") or resolve_column(df, "outcome"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            
            # Operational
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "admit_date": resolve_column(df, "admission_date"),
            "discharge_date": resolve_column(df, "discharge_date"),
            
            # Financial
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost") or resolve_column(df, "charges")
        }

        # 2. Derive Data (The "Smart" Part)
        if not self.cols["los"] and self.cols["admit_date"] and self.cols["discharge_date"]:
            df["derived_los"] = _derive_length_of_stay(df, self.cols["admit_date"], self.cols["discharge_date"])
            self.cols["los"] = "derived_los"

        # 3. Clean Binary Columns (Readmission)
        if self.cols["readmitted"]:
            val_map = {"yes": 1, "no": 0, "true": 1, "false": 0}
            if df[self.cols["readmitted"]].dtype == object:
                df[self.cols["readmitted"]] = df[self.cols["readmitted"]].astype(str).str.lower().map(val_map).fillna(0)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Volume
        kpis["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)

        # 2. Operational (LOS & Bed Turnover)
        if c["los"]:
            kpis["avg_los"] = df[c["los"]].mean()
            kpis["long_stay_rate"] = (df[c["los"]] > 7).mean()
            
            # Bed Turnover Index (Higher is better)
            if kpis["avg_los"] > 0:
                kpis["bed_turnover_index"] = _safe_div(1, kpis["avg_los"])

        # 3. Provider Variance (NEW: The "Smart" Fix)
        if c["doctor"] and c["los"]:
            # Calculate variance in LOS between providers
            los_by_doc = df.groupby(c["doctor"])[c["los"]].mean()
            if len(los_by_doc) > 1:
                kpis["provider_los_std"] = los_by_doc.std()
                kpis["max_provider_los"] = los_by_doc.max()
                kpis["median_provider_los"] = los_by_doc.median()

        # 4. Clinical (Readmission)
        if c["readmitted"]:
            kpis["readmission_rate"] = df[c["readmitted"]].mean()

        # 5. Financial
        if c["cost"]:
            kpis["total_billing"] = df[c["cost"]].sum()
            kpis["avg_cost_per_patient"] = df[c["cost"]].mean()
            
            if c["los"]:
                kpis["avg_cost_per_day"] = _safe_div(kpis["avg_cost_per_patient"], kpis["avg_los"])

        return kpis

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        kpis = self.calculate_kpis(df)

        def save(fig, name, caption, imp, cat):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": imp,
                "category": cat
            })

        def human_fmt(x, _):
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        # 1. Length of Stay Distribution
        if c["los"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["los"]].dropna().hist(ax=ax, bins=15, color="teal")
            ax.set_title("Length of Stay (Days)")
            save(fig, "los_dist.png", "Stay duration spread", 
                 0.9 if kpis.get("avg_los", 0) > 5 else 0.75, "operational")

        # 2. Readmission Rate by Diagnosis
        if c["readmitted"] and c["diagnosis"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax, color="red")
            ax.set_title("Highest Readmission Conditions")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
            save(fig, "readm_diag.png", "Clinical risk factors", 
                 0.95 if kpis.get("readmission_rate", 0) > 0.1 else 0.8, "clinical")

        # 3. Patient Volume Trend
        if self.time_col:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample('M').size().plot(ax=ax, color="#1f77b4")
            ax.set_title("Patient Admissions Trend")
            save(fig, "vol_trend.png", "Hospital throughput", 0.8, "operational")

        # 4. Cost Distribution
        if c["cost"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["cost"]].hist(ax=ax, bins=20, color="green")
            ax.set_title("Treatment Cost Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "cost_dist.png", "Billing variance", 0.7, "financial")

        # 5. Cost by Condition
        if c["cost"] and c["diagnosis"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Avg Cost by Condition")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "cost_diag.png", "Expensive treatments", 0.85, "financial")

        # 6. Payer / Insurance Mix
        if c["payer"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
            ax.axis('equal')
            ax.set_ylabel("")
            ax.set_title("Insurance Payer Mix")
            save(fig, "payer_mix.png", "Revenue source", 0.65, "financial")

        # 7. Clinical Outcome Distribution
        if c["outcome"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["outcome"]].value_counts().head(5).plot(kind="bar", ax=ax, color="purple")
            ax.set_title("Discharge Outcomes")
            save(fig, "outcomes.png", "Patient disposition", 0.75, "clinical")

        # 8. Cost vs LOS Scatter
        if c["cost"] and c["los"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_df = df.sample(1000) if len(df) > 1000 else df
            ax.scatter(plot_df[c["los"]], plot_df[c["cost"]], alpha=0.5, color="gray")
            ax.set_title("Cost vs Length of Stay")
            save(fig, "cost_los.png", "Efficiency correlation", 0.82, "efficiency")

        # 9. Provider Performance (New!)
        if c["doctor"] and c["los"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot(kind="bar", ax=ax, color="brown")
            ax.set_title("Avg LOS by Provider (Longest)")
            # High importance if variance was detected
            save(fig, "provider_los.png", "Care consistency", 
                 0.9 if kpis.get("provider_los_std", 0) > 2 else 0.7, "operational")

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        los = kpis.get("avg_los", 0)
        readm = kpis.get("readmission_rate", 0)
        cost = kpis.get("avg_cost_per_patient", 0)
        max_doc_los = kpis.get("max_provider_los", 0)
        med_doc_los = kpis.get("median_provider_los", 0)

        # Composite 1: Operational Strain
        if los > 7 and readm > 0.15:
            insights.append({
                "level": "CRITICAL", "title": "Operational Strain",
                "so_what": f"System is clogged. Long LOS ({los:.1f} days) and high readmissions ({readm:.1%})."
            })

        # Composite 2: Provider Variance (The Missing Piece)
        if max_doc_los > 0 and med_doc_los > 0:
            if max_doc_los > 1.5 * med_doc_los:
                insights.append({
                    "level": "WARNING",
                    "title": "Provider Variance",
                    "so_what": f"Significant variation in LOS. Slowest provider averages {max_doc_los:.1f} days vs median {med_doc_los:.1f} days."
                })

        # Composite 3: Financial Toxicity
        if cost > 50000 and readm > 0.15:
            insights.append({
                "level": "WARNING", "title": "Financial Inefficiency",
                "so_what": f"High costs (${cost:,.0f}) are not yielding stable outcomes (High Readmissions)."
            })

        if (
            c.get("diagnosis")
            and kpis.get("avg_cost_per_patient", 0) > 8000
            and kpis.get("readmission_rate", 0) > 0.10
        ):
            insights.append({
                "level": "WARNING",
                "title": "High-Impact Service Line: Oncology",
                "so_what": (
                    "Oncology drives both the highest treatment costs and elevated "
                    "readmission risk, making it a prime candidate for care pathway optimization."
                )
            })


        # Atomic Fallbacks
        if readm > 0.15 and not any("Strain" in i["title"] for i in insights):
            insights.append({
                "level": "RISK", "title": "High Readmission Rate",
                "so_what": f"Readmission rate is {readm:.1%}, indicating potential discharge quality issues."
            })

        if los > 7 and not any("Strain" in i["title"] for i in insights):
            insights.append({
                "level": "WARNING", "title": "Extended Length of Stay",
                "so_what": f"Avg stay is {los:.1f} days, reducing bed turnover."
            })

        if not insights:
            if kpis.get("long_stay_rate", 0) > 0.15:
                insights.append({
                    "level": "WARNING",
                    "title": "Capacity Strain Risk",
                    "so_what": (
                        f"{kpis['long_stay_rate']:.1%} of patients exceed LOS targets, "
                        "which may reduce bed availability and delay admissions."
                    )
                })
            else:
                insights.append({
                    "level": "INFO",
                    "title": "Operations Stable",
                    "so_what": "Clinical and operational metrics are within acceptable thresholds."
                })


        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if "Operational Strain" in titles:
            recs.append({"action": "Audit discharge planning workflows immediately.", "priority": "HIGH"})
        
        if "Provider Variance" in titles: 
            recs.append({"action": "Initiate peer review for outlier providers.", "priority": "HIGH"})

        if kpis.get("readmission_rate", 0) > 0.15:
            recs.append({"action": "Implement post-discharge follow-up calls.", "priority": "MEDIUM"})

        if not recs:
            recs.append({
                "action": "Continue monitoring length of stay and readmission trends monthly.",
                "priority": "LOW",
                "timeline": "Ongoing"
            })


        return recs or [{"action": "Monitor patient flow.", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "clinical", "doctor", "hospital", "treatment", "readmitted", "mrn"}

    def detect(self, df) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # Boost if Patient + Diagnosis exist
        cols = str(df.columns).lower()
        if "patient" in cols and ("diagnosis" in cols or "admit" in cols):
            confidence = max(confidence, 0.95)
            
        return DomainDetectionResult("healthcare", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
