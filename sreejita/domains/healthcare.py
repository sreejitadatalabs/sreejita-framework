import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# CONSTANTS
# =====================================================

NEGATIVE_KEYWORDS = {
    "abnormal", "failed", "deceased", "critical",
    "positive", "expired", "severe", "poor"
}

OUTCOME_HINTS = {
    "result", "outcome", "test", "finding", "status", "evaluation"
}


# =====================================================
# HELPERS (PURE FUNCTIONS)
# =====================================================

def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    Derive LOS only when explicit admission & discharge columns
    can be resolved safely.
    """
    df_out = df.copy()

    # Trust explicit LOS if provided
    if resolve_column(df_out, "length_of_stay"):
        return df_out

    admit = (
        resolve_column(df_out, "admission_date")
        or resolve_column(df_out, "admission")
    )
    discharge = (
        resolve_column(df_out, "discharge_date")
        or resolve_column(df_out, "discharge")
    )

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
                "title": f"High Rate of Negative Outcomes in '{col}'",
                "so_what": (
                    f"{rate:.0%} of records show adverse values "
                    f"(e.g., {', '.join(negatives.unique()[:2])})."
                ),
            })

    return insights


# =====================================================
# DOMAIN IMPLEMENTATION
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Defensible healthcare analytics (clinical, financial, operational)"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return (
            resolve_column(df, "patient_id") is not None
            or resolve_column(df, "billing_amount") is not None
        )

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _normalize_binary_columns(df)
        df = _derive_length_of_stay(df)
        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        los = (
            resolve_column(df, "length_of_stay")
            or resolve_column(df, "derived_length_of_stay")
        )
        if los and pd.api.types.is_numeric_dtype(df[los]) and df[los].notna().any():
            kpis["avg_length_of_stay"] = df[los].mean()

        readm = resolve_column(df, "readmitted")
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()

        bill = resolve_column(df, "billing_amount")
        pid = resolve_column(df, "patient_id")
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            if pid:
                kpis["avg_billing_per_patient"] = (
                    df.groupby(pid)[bill].sum().mean()
                )

        if pid:
            kpis["patient_volume"] = df[pid].nunique()

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        def human_fmt(x, _):
            if x >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if x >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))

        # LOS Distribution
        los = (
            resolve_column(df, "length_of_stay")
            or resolve_column(df, "derived_length_of_stay")
        )
        if los and df[los].notna().any():
            p = output_dir / "length_of_stay.png"
            plt.figure(figsize=(6, 4))
            df[los].dropna().hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Patient stay duration distribution"})

        # Billing by Insurance
        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins:
            p = output_dir / "billing_by_insurance.png"
            plt.figure(figsize=(6, 4))
            ax = df.groupby(ins)[bill].sum().sort_values(ascending=False).plot(kind="bar")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.title("Billing by Insurance Provider")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Revenue concentration by payer"})

        # Avg Cost by Diagnosis / Condition
        diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
        if diag and bill:
            p = output_dir / "cost_by_condition.png"
            plt.figure(figsize=(6, 4))
            df.groupby(diag)[bill].mean().sort_values(ascending=False).head(7).plot(kind="barh")
            plt.title("Avg Cost by Condition (Top 7)")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Average treatment cost by condition"})

        return visuals

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        # 1. Categorical clinical risks
        insights.extend(_scan_categorical_risks(df))

        # 2. Doctor deviation (explicit readmission only)
        doc = resolve_column(df, "doctor")
        readm = resolve_column(df, "readmitted")
        if doc and readm and pd.api.types.is_numeric_dtype(df[readm]):
            overall = df[readm].mean()
            grp = df.groupby(doc)[readm].mean()
            worst = grp.idxmax()
            worst_val = grp.max()
            if worst_val > overall * 1.2:
                insights.append({
                    "level": "RISK",
                    "title": f"Worst Performing Doctor: {worst}",
                    "so_what": (
                        f"Readmission rate {worst_val:.1%} "
                        f"vs average {overall:.1%}."
                    ),
                })

        # 3. Financial concentration fallback
        if not insights:
            bill = resolve_column(df, "billing_amount")
            diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
            if bill and diag:
                grp = df.groupby(diag)[bill].sum()
                if not grp.empty:
                    top = grp.idxmax()
                    insights.append({
                        "level": "INFO",
                        "title": f"Highest Cost Driver: {top}",
                        "so_what": f"Accounts for {grp[top]/grp.sum():.0%} of total billing.",
                    })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        for i in self.generate_insights(df, kpis):
            if i["level"] == "RISK":
                recs.append({
                    "action": f"Investigate immediately: {i['title']}",
                    "priority": "HIGH",
                    "timeline": "2–4 weeks",
                })

        if not recs:
            recs.append({
                "action": "Continue routine monitoring",
                "priority": "LOW",
                "timeline": "Ongoing",
            })

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_TOKENS: Set[str] = {
        "patient", "admission", "discharge", "readmitted",
        "diagnosis", "clinical", "physician", "doctor",
        "insurance", "billing", "mortality", "los"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.HEALTHCARE_TOKENS)]
        return DomainDetectionResult(
            domain="healthcare",
            confidence=min(len(hits) / 3, 1.0),
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
