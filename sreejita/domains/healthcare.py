import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# CONSTANTS
# =====================================================

NEGATIVE_KEYWORDS = {
    "abnormal", "failed", "deceased", "critical",
    "positive", "expired", "severe"
}

OUTCOME_HINTS = {"result", "outcome", "test", "finding"}


# =====================================================
# HELPERS
# =====================================================

def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            vals = set(df[col].dropna().astype(str).str.lower().unique())
            if vals and vals.issubset({"yes", "no", "true", "false"}):
                df[col] = df[col].str.lower().map(
                    {"yes": 1, "true": 1, "no": 0, "false": 0}
                )
    return df


def _derive_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    if resolve_column(df, "length_of_stay"):
        return df

    admit = resolve_column(df, "admission_date")
    discharge = resolve_column(df, "discharge_date")

    if admit and discharge:
        a = pd.to_datetime(df[admit], errors="coerce")
        d = pd.to_datetime(df[discharge], errors="coerce")
        los = (d - a).dt.days
        los = los.where(los >= 0)
        if los.notna().any():
            df["derived_length_of_stay"] = los

    return df


def _scan_categorical_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights = []

    for col in df.columns:
        c = col.lower()
        if not any(h in c for h in OUTCOME_HINTS):
            continue
        if df[col].dtype != object:
            continue

        values = df[col].dropna().astype(str).str.lower()
        neg = values[values.isin(NEGATIVE_KEYWORDS)]
        if len(values) == 0:
            continue

        rate = len(neg) / len(values)
        if rate >= 0.15:
            insights.append({
                "level": "RISK" if rate >= 0.25 else "WARNING",
                "title": f"High Rate of Negative Outcomes in {col}",
                "so_what": f"{rate:.0%} of records show adverse outcomes."
            })

    return insights


# =====================================================
# DOMAIN
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return resolve_column(df, "patient_id") is not None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _normalize_binary_columns(df)
        df = _derive_length_of_stay(df)
        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}

        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]) and df[los].notna().any():
            kpis["avg_length_of_stay"] = df[los].mean()

        bill = resolve_column(df, "billing_amount")
        pid = resolve_column(df, "patient_id")
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            if pid:
                kpis["avg_billing_per_patient"] = df.groupby(pid)[bill].sum().mean()

        if pid:
            kpis["patient_volume"] = df[pid].nunique()

        return kpis

    def generate_visuals(self, df: pd.DataFrame, out: Path) -> List[Dict[str, Any]]:
        visuals = []
        out.mkdir(parents=True, exist_ok=True)

        def fmt(x, _):
            if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if x >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and df[los].notna().any():
            p = out / "length_of_stay.png"
            df[los].dropna().hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Patient stay duration distribution"})

        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins:
            p = out / "billing_by_insurance.png"
            ax = df.groupby(ins)[bill].sum().plot(kind="bar")
            ax.yaxis.set_major_formatter(FuncFormatter(fmt))
            plt.title("Billing by Insurance Provider")
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Revenue concentration by payer"})

        return visuals

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        insights.extend(_scan_categorical_risks(df))

        if not insights:
            bill = resolve_column(df, "billing_amount")
            diag = resolve_column(df, "diagnosis")
            if bill and diag:
                grp = df.groupby(diag)[bill].sum()
                top = grp.idxmax()
                insights.append({
                    "level": "INFO",
                    "title": f"Highest Cost Diagnosis: {top}",
                    "so_what": f"Accounts for {grp[top]/grp.sum():.0%} of total billing."
                })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        for i in self.generate_insights(df, kpis):
            if i["level"] == "RISK":
                recs.append({
                    "action": f"Investigate immediately: {i['title']}",
                    "priority": "HIGH",
                    "timeline": "2–4 weeks"
                })
        if not recs:
            recs.append({
                "action": "Continue routine monitoring",
                "priority": "LOW",
                "timeline": "Ongoing"
            })
        return recs


# =====================================================
# DETECTOR
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    HEALTHCARE_COLUMNS = {
        "patient", "admission", "discharge",
        "billing", "insurance", "doctor", "diagnosis"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {c.lower() for c in df.columns}
        hits = [c for c in cols if any(h in c for h in self.HEALTHCARE_COLUMNS)]
        return DomainDetectionResult(
            domain="healthcare",
            confidence=min(len(hits) / 4, 1.0),
            signals={"matched_columns": hits},
        )


def register(registry):
    registry.register(
        name="healthcare",
        domain_cls=HealthcareDomain,
        detector_cls=HealthcareDomainDetector,
    )
