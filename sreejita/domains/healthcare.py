import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# Core imports (Assuming these exist in your framework)
from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# CONSTANTS & CONFIGURATION
# =====================================================

# Strict list of negative keywords to prevent false positives
NEGATIVE_KEYWORDS = {
    "abnormal", "failed", "deceased", "critical",
    "positive", "expired", "severe", "poor"
}

# Only scan columns that likely contain clinical results
OUTCOME_HINTS = {"result", "outcome", "test", "finding", "status", "evaluation"}

# =====================================================
# STATELESS HELPERS (Pure Functions)
# =====================================================

def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely converts Yes/No/True/False to 1/0 without aggressive casting.
    """
    df_out = df.copy()  # Work on copy to avoid mutation side-effects
    for col in df_out.columns:
        if df_out[col].dtype == object:
            # Get unique values to check if it's a binary column
            vals = set(df_out[col].dropna().astype(str).str.lower().unique())
            if vals and vals.issubset({"yes", "no", "true", "false", "y", "n"}):
                df_out[col] = df_out[col].astype(str).str.lower().map(
                    {"yes": 1, "true": 1, "y": 1, "no": 0, "false": 0, "n": 0}
                )
    return df_out

def _derive_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates LOS only if explicit admission/discharge columns are resolved.
    Does NOT fuzzy match column names to avoid calculating on 'Scheduled Date'.
    """
    df_out = df.copy()
    
    # If LOS exists, trust the source data
    if resolve_column(df_out, "length_of_stay"):
        return df_out

    # Resolve dates via the core resolver (Configuration over Guesswork)
    admit = resolve_column(df_out, "admission_date")
    discharge = resolve_column(df_out, "discharge_date")

    if admit and discharge:
        try:
            a = pd.to_datetime(df_out[admit], errors="coerce")
            d = pd.to_datetime(df_out[discharge], errors="coerce")
            
            # Calculate difference
            los = (d - a).dt.days
            
            # Validation: Filter out negative LOS (data errors)
            los = los.where(los >= 0)
            
            if los.notna().any():
                df_out["derived_length_of_stay"] = los
        except Exception:
            pass # Fail safe: Do not crash pipeline on date errors

    return df_out

def _scan_categorical_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans likely outcome columns for high rates of negative keywords.
    """
    insights = []

    for col in df.columns:
        # 1. Guard Clause: Only check relevant columns
        if not any(hint in col.lower() for hint in OUTCOME_HINTS):
            continue
        if df[col].dtype != object:
            continue

        # 2. Analysis
        values = df[col].dropna().astype(str).str.lower()
        if len(values) == 0:
            continue

        negatives = values[values.isin(NEGATIVE_KEYWORDS)]
        rate = len(negatives) / len(values)

        # 3. Thresholding (15% bad outcomes = Risk)
        if rate >= 0.15:
            insights.append({
                "level": "RISK" if rate >= 0.25 else "WARNING",
                "title": f"High Rate of Negative Outcomes in '{col}'",
                "so_what": f"{rate:.0%} of records show adverse values (e.g., {', '.join(negatives.unique()[:2])}).",
                "detected_col": col  # Metadata for downstream use
            })

    return insights

# =====================================================
# DOMAIN LOGIC
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Defensible healthcare analytics (Clinical, Financial, Ops)"

    def validate_data(self, df: pd.DataFrame) -> bool:
        # Minimal requirement: Patient ID or Billing data
        return (resolve_column(df, "patient_id") is not None) or \
               (resolve_column(df, "billing_amount") is not None)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pipeline: Normalize -> Derive. Returns new DF, does not mutate input.
        df = _normalize_binary_columns(df)
        df = _derive_length_of_stay(df)
        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}
        
        # 1. Length of Stay (Prioritize explicit, fallback to derived)
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            kpis["avg_length_of_stay"] = df[los].mean()

        # 2. Readmission (Explicit only - Safety First)
        readm = resolve_column(df, "readmitted")
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()

        # 3. Financials
        bill = resolve_column(df, "billing_amount")
        pid = resolve_column(df, "patient_id")
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            if pid:
                kpis["avg_billing_per_patient"] = df.groupby(pid)[bill].sum().mean()

        if pid:
            kpis["patient_volume"] = df[pid].nunique()

        return kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # Helper for currency formatting
        def human_fmt(x, _):
            if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if x >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        # Visual 1: LOS Distribution
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            p = output_dir / "length_of_stay.png"
            plt.figure(figsize=(6, 4))
            df[los].dropna().hist(bins=15, color='#4A90E2', edgecolor='black')
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Patient stay duration distribution"})

        # Visual 2: Billing by Payer
        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins:
            p = output_dir / "billing_by_insurance.png"
            plt.figure(figsize=(6, 4))
            ax = df.groupby(ins)[bill].sum().sort_values(ascending=False).plot(kind="bar", color='#50E3C2')
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.title("Billing by Insurance Provider")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Revenue concentration by payer"})

        # Visual 3: Cost by Diagnosis (Safe Fallback)
        # Using explicit columns allows us to be defensible.
        diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
        if diag and bill:
            p = output_dir / "cost_by_condition.png"
            plt.figure(figsize=(6, 4))
            df.groupby(diag)[bill].mean().sort_values(ascending=False).head(7).plot(kind="barh", color='#F5A623')
            plt.title("Avg Cost by Condition (Top 7)")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Average treatment cost by condition"})

        return visuals

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        # 1. Risks (Categorical Scan)
        insights.extend(_scan_categorical_risks(df))

        # 2. Financial Concentration (Always Safe)
        bill = resolve_column(df, "billing_amount")
        diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
        
        if bill and diag:
            grp = df.groupby(diag)[bill].sum()
            if not grp.empty:
                top = grp.idxmax()
                share = grp[top] / grp.sum()
                insights.append({
                    "level": "INFO",
                    "title": f"Highest Cost Driver: {top}",
                    "so_what": f"This condition accounts for {share:.1%} of total billing."
                })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        insights = self.generate_insights(df, kpis)

        for i in insights:
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
    
    # Specific terms only. No "date" or generic words.
    HEALTHCARE_TOKENS = {
        "patient", "admission", "discharge", "readmitted",
        "diagnosis", "triage", "clinical", "physician", "doctor",
        "insurance", "billing", "mortality", "los"
    }

    def detect(self, df) -> DomainDetectionResult:
        # High-confidence matching only
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.HEALTHCARE_TOKENS)]
        
        return DomainDetectionResult(
            domain="healthcare",
            confidence=min(len(hits) / 3, 1.0),
            signals={"matched_columns": hits},
        )

def register(registry):
    registry.register(
        name="healthcare",
        domain_cls=HealthcareDomain,
        detector_cls=HealthcareDomainDetector,
    )
