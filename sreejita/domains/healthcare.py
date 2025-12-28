import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
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


def _detect_time_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["admission", "visit", "date", "discharge"]):
            try:
                pd.to_datetime(df[c].dropna().iloc[0])
                return c
            except Exception:
                continue
    return None


def _derive_los(df, admit, discharge):
    a = pd.to_datetime(df[admit], errors="coerce")
    d = pd.to_datetime(df[discharge], errors="coerce")
    los = (d - a).dt.days
    return los.where(los >= 0)


# =====================================================
# HEALTHCARE DOMAIN — v3.6.1 GOLD
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Clinical, Operational & Financial Healthcare Intelligence"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        signals = ["patient", "admission", "diagnosis", "cost", "readmit", "los"]
        cols = " ".join(df.columns).lower()
        return sum(s in cols for s in signals) >= 2

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)

        self.cols = {
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "mrn"),
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "condition"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "readmission"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "payer": resolve_column(df, "insurance") or resolve_column(df, "payer"),
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "admit": resolve_column(df, "admission_date"),
            "gender": resolve_column(df, "gender") or resolve_column(df, "sex"),
            "age": resolve_column(df, "age"),
            "outcome": resolve_column(df, "discharge_status") or resolve_column(df, "outcome"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),
            "discharge": resolve_column(df, "discharge_date"),
            "cost": resolve_column(df, "billing_amount") or resolve_column(df, "total_cost"),
        }

        # Derive LOS if missing
        if not self.cols["los"] and self.cols["admit"] and self.cols["discharge"]:
            df["derived_los"] = _derive_los(df, self.cols["admit"], self.cols["discharge"])
            self.cols["los"] = "derived_los"

        # Normalize readmission
        if self.cols["readmitted"] and df[self.cols["readmitted"]].dtype == object:
            df[self.cols["readmitted"]] = (
                df[self.cols["readmitted"]]
                .astype(str).str.lower()
                .map({"yes": 1, "no": 0, "true": 1, "false": 0})
                .fillna(0)
            )

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df):
        c = self.cols
        k = {}

        k["total_patients"] = df[c["pid"]].nunique() if c["pid"] else len(df)

        if c["los"]:
            k["avg_los"] = df[c["los"]].mean()
            k["long_stay_rate"] = (df[c["los"]] > 7).mean()
            k["bed_turnover_index"] = _safe_div(1, k["avg_los"])

        if c["readmitted"]:
            k["readmission_rate"] = df[c["readmitted"]].mean()

        if c["doctor"] and c["los"]:
            los_doc = df.groupby(c["doctor"])[c["los"]].mean()
            if len(los_doc) > 1:
                k["max_provider_los"] = los_doc.max()
                k["median_provider_los"] = los_doc.median()
                k["provider_los_std"] = los_by_doc.std()
                
        if c["cost"]:
            k["avg_cost_per_patient"] = df[c["cost"]].mean()
            k["total_billing"] = df[c["cost"]].sum()
            
        if c["los"]:
            k["avg_cost_per_day"] = _safe_div(kpis["avg_cost_per_patient"], kpis["avg_los"])


        return k

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(self, df, output_dir: Path):
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        k = self.calculate_kpis(df)

        def save(fig, name, caption, imp):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": imp,
                "category": "healthcare"
            })

        fmt = FuncFormatter(lambda x, _: f"{x/1e3:.0f}K")

        # 1. LOS Distribution
        if c["los"]:
            fig, ax = plt.subplots()
            df[c["los"]].dropna().hist(bins=10, ax=ax)
            ax.set_title("Length of Stay Distribution")
            save(fig, "los_dist.png", "Patient stay duration", 0.9)

        # 2. Readmission by Diagnosis
        if c["readmitted"] and c["diagnosis"]:
            fig, ax = plt.subplots()
            df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot.barh(ax=ax)
            ax.set_title("Highest Readmission Conditions")
            save(fig, "readmit_diag.png", "Clinical risk drivers", 0.95)

        # 3. Cost by Diagnosis
        if c["cost"] and c["diagnosis"]:
            fig, ax = plt.subplots()
            df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot.bar(ax=ax)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_title("Avg Cost by Condition")
            save(fig, "cost_diag.png", "Cost drivers", 0.9)

        # 4. Volume Trend
        if self.time_col:
            fig, ax = plt.subplots()
            df.set_index(self.time_col).resample("M").size().plot(ax=ax)
            ax.set_title("Patient Volume Trend")
            save(fig, "volume_trend.png", "Demand pressure", 0.8)

        # 5. Cost vs LOS
        if c["cost"] and c["los"]:
            fig, ax = plt.subplots()
            ax.scatter(df[c["los"]], df[c["cost"]], alpha=0.4)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_title("Cost vs Length of Stay")
            save(fig, "cost_vs_los.png", "Efficiency correlation", 0.85)

        # 6. Provider LOS Variance
        if c["doctor"] and c["los"]:
            fig, ax = plt.subplots()
            df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot.bar(ax=ax)
            ax.set_title("Avg LOS by Provider")
            save(fig, "provider_los.png", "Care consistency", 0.85)

        # 7. Cost Distribution
        if c["cost"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["cost"]].hist(ax=ax, bins=20, color="green")
            ax.set_title("Treatment Cost Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "cost_dist.png", "Billing variance", 0.7, "financial")

       
        # 8. Payer / Insurance Mix
        if c["payer"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["payer"]].value_counts().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
            ax.axis('equal')
            ax.set_ylabel("")
            ax.set_title("Insurance Payer Mix")
            save(fig, "payer_mix.png", "Revenue source", 0.65, "financial")

        # 9. Clinical Outcome Distribution
        if c["outcome"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["outcome"]].value_counts().head(5).plot(kind="bar", ax=ax, color="purple")
            ax.set_title("Discharge Outcomes")
            save(fig, "outcomes.png", "Patient disposition", 0.75, "clinical")

        # Rank & cap
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- COMPOSITE INSIGHTS ----------------

    def generate_insights(self, df, k):
        insights = []

        los = k.get("avg_los", 0)
        long_rate = k.get("long_stay_rate", 0)
        readm = k.get("readmission_rate", 0)
        cost = k.get("avg_cost_per_patient", 0)

        # 1. Capacity Strain
        if long_rate > 0.25:
            insights.append({
                "level": "CRITICAL",
                "title": "Capacity Strain",
                "so_what": f"{long_rate:.1%} of patients exceed LOS targets, limiting bed availability."
            })
        elif long_rate > 0.15:
            insights.append({
                "level": "WARNING",
                "title": "Extended Length of Stay",
                "so_what": f"Long stay rate at {long_rate:.1%} risks throughput."
            })

        # 2. Financial Toxicity
        if cost > 50000 and readm > 0.15:
            insights.append({
                "level": "WARNING",
                "title": "Financial Inefficiency",
                "so_what": "High costs are not translating into stable outcomes."
            })

        # 3. Provider Variance
        if k.get("max_provider_los") and k.get("median_provider_los"):
            if k["max_provider_los"] > 1.5 * k["median_provider_los"]:
                insights.append({
                    "level": "WARNING",
                    "title": "Provider Variance",
                    "so_what": "LOS variation indicates inconsistent care pathways."
                })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Operations Stable",
                "so_what": "Clinical and operational metrics within expected ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df, k):
        recs = []

        if k.get("long_stay_rate", 0) > 0.15:
            recs.append({
                "action": "Audit discharge planning for long-stay patients.",
                "priority": "HIGH"
            })

        if k.get("readmission_rate", 0) > 0.15:
            recs.append({
                "action": "Strengthen post-discharge follow-ups within 48 hours.",
                "priority": "MEDIUM"
            })

        return recs or [{
            "action": "Continue monitoring patient flow and outcomes.",
            "priority": "LOW"
        }]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {"patient", "admission", "diagnosis", "readmission", "hospital"}

    def detect(self, df):
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits) / 3, 1.0)
        return DomainDetectionResult("healthcare", confidence, {"matched_columns": hits})


def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
