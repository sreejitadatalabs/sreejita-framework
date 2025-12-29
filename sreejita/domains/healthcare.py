import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["admission", "visit", "date", "discharge", "month"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:3], errors="raise")
                return c
            except Exception:
                continue
    return None


def _derive_los(df, admit_col, discharge_col):
    try:
        a = pd.to_datetime(df[admit_col], errors="coerce")
        d = pd.to_datetime(df[discharge_col], errors="coerce")
        los = (d - a).dt.days
        return los.where(los >= 0)
    except Exception:
        return None


def detect_dataset_shape(cols: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Detect dataset grain to avoid dataset bias.
    """
    if cols.get("volume"):
        grain = "aggregated"
    elif cols.get("pid"):
        grain = "patient"
    else:
        grain = "unknown"

    return {
        "grain": grain,
        "has_cost": bool(cols.get("cost")),
        "has_provider": bool(cols.get("doctor")),
    }


# =====================================================
# HEALTHCARE DOMAIN — UNIVERSAL
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Universal Healthcare Intelligence (Clinical, Operational, Financial)"

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)

        self.cols = {
            # Identity
            "pid": resolve_column(df, "patient_id") or resolve_column(df, "mrn"),

            # Clinical
            "diagnosis": resolve_column(df, "diagnosis") or resolve_column(df, "condition"),
            "readmitted": resolve_column(df, "readmitted") or resolve_column(df, "readmission"),
            "outcome": resolve_column(df, "discharge_status") or resolve_column(df, "outcome"),
            "doctor": resolve_column(df, "doctor") or resolve_column(df, "provider"),

            # Operational
            "los": resolve_column(df, "length_of_stay") or resolve_column(df, "los"),
            "admit": resolve_column(df, "admission_date"),
            "discharge": resolve_column(df, "discharge_date"),

            # Financial
            "cost": resolve_column(df, "billing_amount") \
                    or resolve_column(df, "total_cost") \
                    or resolve_column(df, "charges"),

            # Aggregated
            "volume": resolve_column(df, "total_patients") or resolve_column(df, "visits"),
        }

        self.shape = detect_dataset_shape(self.cols)

        # Derive LOS if missing
        if not self.cols["los"] and self.cols["admit"] and self.cols["discharge"]:
            df["derived_los"] = _derive_los(df, self.cols["admit"], self.cols["discharge"])
            self.cols["los"] = "derived_los"

        # Normalize readmission
        if self.cols["readmitted"] and df[self.cols["readmitted"]].dtype == object:
            mapping = {
                "yes": 1, "no": 0, "true": 1, "false": 0,
                "1": 1, "0": 0, "y": 1, "n": 0,
            }
            df[self.cols["readmitted"]] = (
                df[self.cols["readmitted"]]
                .astype(str).str.lower()
                .map(mapping)
                .fillna(0)
            )

        return df

    # -------------------------------------------------
    # KPIs
    # -------------------------------------------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        c = self.cols
        k: Dict[str, Any] = {}

        # Volume
        if self.shape["grain"] == "aggregated" and c["volume"]:
            k["total_patients"] = df[c["volume"]].sum()
            k["is_aggregated"] = True
        elif c["pid"]:
            k["total_patients"] = df[c["pid"]].nunique()
            k["is_aggregated"] = False
        else:
            k["total_patients"] = len(df)
            k["is_aggregated"] = False

        # LOS
        if c["los"]:
            k["avg_los"] = df[c["los"]].mean()
            benchmark = df[c["los"]].median() * 1.5 if len(df) > 10 else 7
            k["benchmark_los"] = benchmark
            k["long_stay_rate"] = (df[c["los"]] > benchmark).mean()
            if k["avg_los"] > 0:
                k["bed_turnover_index"] = _safe_div(1, k["avg_los"])

        # Provider variance (patient-level only)
        if self.shape["grain"] == "patient" and c["doctor"] and c["los"]:
            los_by_doc = df.groupby(c["doctor"])[c["los"]].mean()
            if len(los_by_doc) > 1:
                k["provider_los_std"] = los_by_doc.std()
                k["max_provider_los"] = los_by_doc.max()
                k["median_provider_los"] = los_by_doc.median()

        # Readmissions
        if c["readmitted"]:
            k["readmission_rate"] = df[c["readmitted"]].mean()

        # Financials
        if c["cost"]:
            k["total_billing"] = df[c["cost"]].sum()
            k["avg_cost_per_patient"] = df[c["cost"]].mean()
            k["benchmark_cost"] = df[c["cost"]].median() * 2 if len(df) > 10 else 50000
            if c["los"]:
                k["avg_cost_per_day"] = _safe_div(k["avg_cost_per_patient"], k["avg_los"])

        return k

    # -------------------------------------------------
    # VISUALS (9 CANDIDATES → ORCHESTRATOR CAPS TO 4)
    # -------------------------------------------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        k = self.calculate_kpis(df)

        def save(fig, name, caption, importance):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": importance,
            })

        money_fmt = FuncFormatter(lambda x, _: f"{x/1e3:.0f}K")

        # 1. LOS distribution
        if c["los"]:
            fig, ax = plt.subplots()
            df[c["los"]].dropna().hist(ax=ax, bins=10)
            ax.set_title("Length of Stay Distribution")
            save(fig, "los_dist.png", "Length of stay spread", 0.95)

        # 2. Admissions trend
        if self.time_col:
            fig, ax = plt.subplots()
            df.set_index(pd.to_datetime(df[self.time_col])).resample("M").size().plot(ax=ax)
            ax.set_title("Patient Admissions Trend")
            save(fig, "admissions_trend.png", "Demand trend over time", 0.9)

        # 3. Cost vs LOS
        if c["cost"] and c["los"]:
            fig, ax = plt.subplots()
            ax.scatter(df[c["los"]], df[c["cost"]], alpha=0.4)
            ax.yaxis.set_major_formatter(money_fmt)
            ax.set_title("Cost vs Length of Stay")
            save(fig, "cost_vs_los.png", "Efficiency relationship", 0.88)

        # 4. Readmission by diagnosis
        if c["readmitted"] and c["diagnosis"]:
            fig, ax = plt.subplots()
            df.groupby(c["diagnosis"])[c["readmitted"]].mean().nlargest(5).plot(kind="barh", ax=ax)
            ax.set_title("Highest Readmission Conditions")
            save(fig, "readmit_diag.png", "Clinical risk drivers", 0.92)

        # 5. Cost by diagnosis
        if c["cost"] and c["diagnosis"]:
            fig, ax = plt.subplots()
            df.groupby(c["diagnosis"])[c["cost"]].mean().nlargest(5).plot(kind="bar", ax=ax)
            ax.yaxis.set_major_formatter(money_fmt)
            ax.set_title("Avg Cost by Condition")
            save(fig, "cost_diag.png", "Cost drivers", 0.9)

        # 6. Provider LOS variance
        if c["doctor"] and c["los"] and not k.get("is_aggregated"):
            fig, ax = plt.subplots()
            df.groupby(c["doctor"])[c["los"]].mean().nlargest(10).plot(kind="bar", ax=ax)
            ax.set_title("Avg LOS by Provider")
            save(fig, "provider_los.png", "Care variability", 0.85)

        # 7. Cost distribution
        if c["cost"]:
            fig, ax = plt.subplots()
            df[c["cost"]].hist(ax=ax, bins=20)
            ax.xaxis.set_major_formatter(money_fmt)
            ax.set_title("Cost Distribution")
            save(fig, "cost_dist.png", "Billing spread", 0.8)

        # 8. Outcome distribution
        if c["outcome"]:
            fig, ax = plt.subplots()
            df[c["outcome"]].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Discharge Outcomes")
            save(fig, "outcomes.png", "Patient disposition", 0.75)

        # 9. Bed turnover proxy
        if k.get("bed_turnover_index"):
            fig, ax = plt.subplots()
            ax.bar(["Bed Turnover Index"], [k["bed_turnover_index"]])
            ax.set_title("Bed Turnover Index")
            save(fig, "bed_turnover.png", "Capacity utilization", 0.8)

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals

    # -------------------------------------------------
    # INSIGHTS (ATOMIC + COMPOSITE)
    # -------------------------------------------------

    def generate_insights(self, df: pd.DataFrame, k: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        # Composite: Capacity strain
        if k.get("long_stay_rate", 0) > 0.25:
            insights.append({
                "level": "CRITICAL",
                "title": "Operational Strain",
                "so_what": f"{k['long_stay_rate']:.1%} of patients exceed LOS targets, limiting capacity.",
            })

        # Composite: Financial inefficiency
        if k.get("avg_cost_per_patient", 0) > k.get("benchmark_cost", 50000) and k.get("readmission_rate", 0) > 0.15:
            insights.append({
                "level": "WARNING",
                "title": "Financial Inefficiency",
                "so_what": "High treatment costs are not translating into stable outcomes.",
            })

        # Atomic risks
        if k.get("readmission_rate", 0) > 0.15:
            insights.append({
                "level": "RISK",
                "title": "High Readmission Rate",
                "so_what": f"Readmission rate is {k['readmission_rate']:.1%}.",
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Operations Stable",
                "so_what": "Clinical and operational metrics are within expected ranges.",
            })

        return insights

    # -------------------------------------------------
    # RECOMMENDATIONS
    # -------------------------------------------------

    def generate_recommendations(self, df: pd.DataFrame, k: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if k.get("long_stay_rate", 0) > 0.25:
            recs.append({
                "action": "Audit discharge planning and reduce long-stay cases.",
                "priority": "HIGH",
                "timeline": "Immediate",
            })

        if k.get("readmission_rate", 0) > 0.15:
            recs.append({
                "action": "Strengthen post-discharge follow-up protocols.",
                "priority": "MEDIUM",
                "timeline": "30–60 days",
            })

        return recs or [{
            "action": "Continue monitoring clinical and operational KPIs.",
            "priority": "LOW",
            "timeline": "Ongoing",
        }]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    TOKENS = {
        "patient", "admission", "diagnosis", "clinical",
        "hospital", "doctor", "treatment", "readmission"
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits) / 3, 1.0)
        return DomainDetectionResult("healthcare", confidence, {"matched_columns": hits})


def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
