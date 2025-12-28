import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# CONSTANTS & HELPERS (DOMAIN-SPECIFIC)
# =====================================================

NEGATIVE_KEYWORDS = {
    "abnormal", "failed", "deceased", "critical",
    "positive", "expired", "severe", "poor"
}

OUTCOME_HINTS = {
    "result", "outcome", "test", "finding", "status", "evaluation"
}


def _safe_div(n, d):
    """Safely divides n by d."""
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Yes/No-like categorical columns to 1/0.
    Keeps preprocessing deterministic and reversible.
    """
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].dtype == object:
            vals = set(df_out[col].dropna().astype(str).str.lower().unique())
            if vals and vals.issubset({"yes", "no", "true", "false", "y", "n"}):
                df_out[col] = df_out[col].astype(str).str.lower().map(
                    {"yes": 1, "true": 1, "y": 1, "no": 0, "false": 0, "n": 0}
                )
    return df_out


def _derive_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Length of Stay (LOS) if not explicitly provided.
    Negative or invalid date ranges are discarded safely.
    """
    df_out = df.copy()

    if resolve_column(df_out, "length_of_stay"):
        return df_out

    admit = resolve_column(df_out, "admission_date") or resolve_column(df_out, "admission")
    discharge = resolve_column(df_out, "discharge_date") or resolve_column(df_out, "discharge")

    if admit and discharge:
        try:
            a = pd.to_datetime(df_out[admit], errors="coerce")
            d = pd.to_datetime(df_out[discharge], errors="coerce")
            los = (d - a).dt.days
            los = los.where(los >= 0)
            if los.notna().any():
                df_out["derived_length_of_stay"] = los
        except Exception:
            pass

    return df_out


def _scan_categorical_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scan outcome-related text columns for adverse clinical signals.
    Heuristic only — not a diagnostic system.
    """
    insights: List[Dict[str, Any]] = []

    for col in df.columns:
        if not any(h in col.lower() for h in OUTCOME_HINTS):
            continue
        if df[col].dtype != object:
            continue

        values = df[col].dropna().astype(str).str.lower()
        if values.empty:
            continue

        negatives = values[values.isin(NEGATIVE_KEYWORDS)]
        rate = len(negatives) / len(values)

        if rate >= 0.15:
            insights.append({
                "level": "RISK" if rate >= 0.25 else "WARNING",
                "title": f"High Adverse Outcomes in '{col}'",
                "so_what": (
                    f"{rate:.0%} of records show adverse values "
                    f"(e.g., {', '.join(list(negatives.unique())[:2])})."
                ),
            })

    return insights


# =====================================================
# HEALTHCARE DOMAIN
# =====================================================

class HealthcareDomain(BaseDomain):
    """
    Healthcare analytics covering:
    - Clinical outcomes
    - Operational efficiency
    - Financial exposure
    """

    name = "healthcare"
    description = "Defensible healthcare analytics (clinical, financial, operational)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Minimal validation to activate healthcare domain.
        """
        return (
            resolve_column(df, "patient_id") is not None
            or resolve_column(df, "billing_amount") is not None
            or resolve_column(df, "diagnosis") is not None
        )

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _normalize_binary_columns(df)
        df = _derive_length_of_stay(df)
        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        KPI calculation with safe guards.
        Benchmarks are illustrative defaults, not regulatory thresholds.
        """
        kpis: Dict[str, Any] = {}

        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        readm = resolve_column(df, "readmitted")
        bill = resolve_column(df, "billing_amount") or resolve_column(df, "cost")
        pid = resolve_column(df, "patient_id")

        # Operational
        if los and pd.api.types.is_numeric_dtype(df[los]) and df[los].notna().any():
            kpis["avg_length_of_stay"] = df[los].mean()
            kpis["target_avg_los"] = 5.0  # default benchmark

        # Clinical
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()
            kpis["target_readmission_rate"] = 0.10

        # Financial
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            kpis["avg_treatment_cost"] = df[bill].mean()

            if pid:
                kpis["avg_billing_per_patient"] = (
                    df.groupby(pid)[bill].sum().mean()
                )

        # Volume
        kpis["patient_volume"] = (
            df[pid].nunique() if pid else len(df)
        )

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        def human_fmt(x, _):
            if x >= 1_000_000:
                return f"{x / 1_000_000:.1f}M"
            if x >= 1_000:
                return f"{x / 1_000:.0f}K"
            return str(int(x))

        # 1️⃣ Length of Stay Distribution
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]) and df[los].notna().any():
            fig, ax = plt.subplots(figsize=(7, 4))
            df[los].dropna().hist(ax=ax, bins=15, edgecolor="white")
            ax.set_title("Length of Stay Distribution")
            ax.set_xlabel("Days")

            p = output_dir / "length_of_stay.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)

            visuals.append({"path": str(p), "caption": "Patient stay duration distribution"})

        # 2️⃣ Patient Volume Trend
        date_col = (
            resolve_column(df, "admission_date")
            or resolve_column(df, "visit_date")
            or resolve_column(df, "date")
        )

        if date_col:
            try:
                dfx = df.copy()
                dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
                dfx = dfx.dropna(subset=[date_col])

                if not dfx.empty:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    dfx.set_index(date_col).resample("M").size().plot(ax=ax)
                    ax.set_title("Patient Volume Trend")
                    ax.set_ylabel("Visits")

                    p = output_dir / "patient_volume_trend.png"
                    fig.savefig(p, bbox_inches="tight")
                    plt.close(fig)

                    visuals.append({"path": str(p), "caption": "Monthly patient volume trend"})
            except Exception:
                pass

        # 3️⃣ Cost Distribution
        bill = resolve_column(df, "billing_amount") or resolve_column(df, "cost")
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            fig, ax = plt.subplots(figsize=(7, 4))
            df[bill].dropna().hist(ax=ax, bins=20, edgecolor="white")
            ax.set_title("Treatment Cost Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))

            p = output_dir / "cost_distribution.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)

            visuals.append({"path": str(p), "caption": "Distribution of treatment costs"})

        # 4️⃣ Cost by Condition
        diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
        if diag and bill and pd.api.types.is_numeric_dtype(df[bill]):
            fig, ax = plt.subplots(figsize=(7, 4))
            (
                df.groupby(diag)[bill]
                .mean()
                .sort_values(ascending=False)
                .head(7)
                .plot(kind="barh", ax=ax)
            )
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            ax.set_title("Avg Cost by Condition (Top 7)")

            p = output_dir / "cost_by_condition.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)

            visuals.append({"path": str(p), "caption": "Average treatment cost by condition"})

        # 5️⃣ Readmission by Condition
        readm = resolve_column(df, "readmitted")
        if readm and diag and pd.api.types.is_numeric_dtype(df[readm]):
            fig, ax = plt.subplots(figsize=(7, 4))
            (
                df.groupby(diag)[readm]
                .mean()
                .sort_values(ascending=False)
                .head(7)
                .plot(kind="bar", ax=ax)
            )
            ax.set_title("Readmission Rate by Condition")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
            plt.xticks(rotation=45, ha="right")

            p = output_dir / "readmission_by_condition.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)

            visuals.append({"path": str(p), "caption": "Readmission rate by condition"})

        return visuals[:6]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        insights.extend(_scan_categorical_risks(df))

        composite = self.generate_composite_insights(df, kpis) if len(df) > 30 else []
        dominant_titles = {i["title"] for i in composite if i["level"] in {"RISK", "WARNING"}}

        suppress_los = "Operational Strain Detected" in dominant_titles
        suppress_cost = "Financial Toxicity Risk" in dominant_titles

        los = kpis.get("avg_length_of_stay")
        readm = kpis.get("readmission_rate")
        cost = kpis.get("avg_treatment_cost")

        if los is not None and not suppress_los and los > 7:
            insights.append({
                "level": "WARNING",
                "title": "Extended Length of Stay",
                "so_what": f"Average LOS is {los:.1f} days."
            })

        if readm is not None and readm > 0.15:
            insights.append({
                "level": "RISK",
                "title": "High Readmission Rate",
                "so_what": f"Readmission rate is {readm:.1%}."
            })

        if cost is not None and not suppress_cost and cost > 50000:
            insights.append({
                "level": "WARNING",
                "title": "High Treatment Cost",
                "so_what": f"Average treatment cost is {cost:,.0f}."
            })

        insights += composite

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Healthcare Operations Stable",
                "so_what": "Clinical, operational, and financial indicators are within acceptable limits."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS ----------------

    def generate_composite_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        los = kpis.get("avg_length_of_stay")
        readm = kpis.get("readmission_rate")
        cost = kpis.get("avg_treatment_cost")

        if los and readm and los > 7 and readm > 0.15:
            insights.append({
                "level": "RISK",
                "title": "Operational Strain Detected",
                "so_what": (
                    f"Extended LOS ({los:.1f} days) with high readmissions "
                    f"({readm:.1%}) indicates capacity strain."
                )
            })

        if cost and readm and cost > 50000 and readm > 0.15:
            insights.append({
                "level": "WARNING",
                "title": "Financial Toxicity Risk",
                "so_what": (
                    f"High costs ({cost:,.0f}) with poor outcomes "
                    f"(readmissions {readm:.1%})."
                )
            })

        doc = resolve_column(df, "doctor")
        readm_col = resolve_column(df, "readmitted")

        if doc and readm_col and pd.api.types.is_numeric_dtype(df[readm_col]):
            if df[doc].nunique() > 2:
                overall = df[readm_col].mean()
                grp = df.groupby(doc)[readm_col].mean()
                worst_doc = grp.idxmax()
                worst_val = grp.max()

                if worst_val > overall * 1.3 and overall > 0:
                    insights.append({
                        "level": "RISK",
                        "title": "Provider Performance Variance",
                        "so_what": (
                            f"Dr. {worst_doc} has a readmission rate of {worst_val:.1%}, "
                            f"higher than average ({overall:.1%})."
                        )
                    })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        composite = self.generate_composite_insights(df, kpis) if len(df) > 30 else []
        titles = [i["title"] for i in composite]

        if "Operational Strain Detected" in titles:
            return [{
                "action": "Audit discharge planning and bed management workflows",
                "priority": "HIGH",
                "timeline": "Immediate"
            }]

        if "Financial Toxicity Risk" in titles:
            return [{
                "action": "Review cost-efficiency of treatment protocols",
                "priority": "HIGH",
                "timeline": "This Month"
            }]

        if "Provider Performance Variance" in titles:
            return [{
                "action": "Initiate peer review for outlier providers",
                "priority": "HIGH",
                "timeline": "This Quarter"
            }]

        return [{
            "action": "Continue monitoring healthcare performance indicators",
            "priority": "LOW",
            "timeline": "Ongoing"
        }]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_TOKENS: Set[str] = {
        "patient", "admission", "discharge", "readmitted",
        "diagnosis", "clinical", "physician", "doctor",
        "insurance", "billing", "mortality", "los", "medical"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.HEALTHCARE_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        STRONG_SIGNALS = {"diagnosis", "clinical", "patient", "readmitted", "physician"}
        if any(t in c for c in cols for t in STRONG_SIGNALS):
            confidence = max(confidence, 0.9)

        return DomainDetectionResult(
            domain="healthcare",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="healthcare",
        domain_cls=HealthcareDomain,
        detector_cls=HealthcareDomainDetector,
    )
