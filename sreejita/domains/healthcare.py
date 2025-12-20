import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# CONSTANTS & HELPERS
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
    """Standardizes Yes/No columns to 1/0 for easier analysis."""
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
    """Derives Length of Stay (LOS) from admission/discharge dates if missing."""
    df_out = df.copy()

    # Trust explicit LOS if provided
    if resolve_column(df_out, "length_of_stay"):
        return df_out

    admit = resolve_column(df_out, "admission_date") or resolve_column(df_out, "admission")
    discharge = resolve_column(df_out, "discharge_date") or resolve_column(df_out, "discharge")

    if admit and discharge:
        try:
            a = pd.to_datetime(df_out[admit], errors="coerce")
            d = pd.to_datetime(df_out[discharge], errors="coerce")
            los = (d - a).dt.days
            los = los.where(los >= 0) # Remove negative dates
            if los.notna().any():
                df_out["derived_length_of_stay"] = los
        except Exception:
            pass

    return df_out

def _scan_categorical_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Scans text columns for medical keywords indicating negative outcomes."""
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
                    f"(e.g., {', '.join(negatives.unique()[:2])})."
                ),
            })

    return insights


# =====================================================
# HEALTHCARE DOMAIN (v3.0 - ENTERPRISE INTELLIGENCE)
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Defensible healthcare analytics (clinical, financial, operational)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
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
        kpis: Dict[str, Any] = {}

        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        readm = resolve_column(df, "readmitted")
        bill = resolve_column(df, "billing_amount")
        pid = resolve_column(df, "patient_id")

        # 1. Operational: Length of Stay
        if los and pd.api.types.is_numeric_dtype(df[los]) and df[los].notna().any():
            kpis["avg_length_of_stay"] = df[los].mean()
            kpis["target_avg_los"] = 5.0 # Benchmark target

        # 2. Clinical: Readmission Rate
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()
            kpis["target_readmission_rate"] = 0.10 # Benchmark target

        # 3. Financial: Billing
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            kpis["avg_billing"] = df[bill].mean()
            
            if pid:
                # Billing per patient (handling multiple visits)
                kpis["avg_billing_per_patient"] = df.groupby(pid)[bill].sum().mean()

        # 4. Volume
        if pid:
            kpis["patient_volume"] = df[pid].nunique()
        else:
            kpis["patient_volume"] = len(df)

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        kpis = self.calculate_kpis(df)

        def human_fmt(x, _):
            if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if x >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        # -------- Visual 1: LOS Distribution --------
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and df[los].notna().any() and pd.api.types.is_numeric_dtype(df[los]):
            p = output_dir / "length_of_stay.png"
            plt.figure(figsize=(7, 4))
            df[los].dropna().hist(bins=15, color="#1f77b4", edgecolor='white')
            plt.title("Length of Stay Distribution")
            plt.xlabel("Days")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Patient stay duration distribution"})

        # -------- Visual 2: Billing by Insurance --------
        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins and pd.api.types.is_numeric_dtype(df[bill]):
            p = output_dir / "billing_by_insurance.png"
            plt.figure(figsize=(7, 4))
            ax = df.groupby(ins)[bill].sum().sort_values(ascending=False).head(7).plot(kind="bar", color="#2ca02c")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.title("Billing by Insurance Provider")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Revenue concentration by payer"})

        # -------- Visual 3: Cost by Condition --------
        diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
        if diag and bill and pd.api.types.is_numeric_dtype(df[bill]):
            p = output_dir / "cost_by_condition.png"
            plt.figure(figsize=(7, 4))
            df.groupby(diag)[bill].mean().sort_values(ascending=False).head(7).plot(kind="barh", color="#d62728")
            plt.title("Avg Cost by Condition (Top 7)")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Average treatment cost by condition"})

        return visuals[:4]

    # ---------------- ATOMIC INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        # 1. Categorical clinical risks (Text Scanning)
        insights.extend(_scan_categorical_risks(df))

        readm_rate = kpis.get("readmission_rate")
        avg_los = kpis.get("avg_length_of_stay")

        # 2. Readmission Insight
        if readm_rate is not None:
            if readm_rate > 0.15:
                insights.append({
                    "level": "RISK",
                    "title": "High Readmission Rate",
                    "so_what": f"Rate is {readm_rate:.1%}, significantly above the 10% benchmark."
                })
            elif readm_rate > 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Elevated Readmissions",
                    "so_what": f"Rate is {readm_rate:.1%}."
                })

        # 3. LOS Insight
        if avg_los is not None and avg_los > 7.0:
            insights.append({
                "level": "INFO",
                "title": "Extended Length of Stay",
                "so_what": f"Average LOS is {avg_los:.1f} days, indicating complex caseloads."
            })

        # === CALL COMPOSITE LAYER (v3.0) ===
        insights += self.generate_composite_insights(df, kpis)

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Clinical Operations Stable",
                "so_what": "Key metrics (LOS, Readmission, Billing) are within normal ranges."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (HEALTHCARE v3.0) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Healthcare v3 Composite Intelligence Layer.
        Connects Clinical outcomes with Financial data.
        """
        insights: List[Dict[str, Any]] = []

        readm_rate = kpis.get("readmission_rate")
        avg_los = kpis.get("avg_length_of_stay")
        avg_cost = kpis.get("avg_billing")
        
        # 1. Operational Strain: High LOS + High Readmission
        if avg_los is not None and readm_rate is not None:
            if avg_los > 7 and readm_rate > 0.15:
                insights.append({
                    "level": "RISK",
                    "title": "Operational Strain (High LOS + Readmissions)",
                    "so_what": (
                        f"Patients are staying long ({avg_los:.1f} days) yet returning frequently "
                        f"({readm_rate:.1%}). Review discharge planning protocols."
                    )
                })

        # 2. "High Cost, Poor Outcome" (Financial Toxicity)
        if avg_cost is not None and readm_rate is not None:
            # Thresholds are illustrative; ideally derived from dataset
            if avg_cost > 15000 and readm_rate > 0.15:
                insights.append({
                    "level": "WARNING",
                    "title": "High Cost / High Readmission Cluster",
                    "so_what": (
                        f"Average treatment cost is high ({int(avg_cost)}) but readmission "
                        f"rates remain elevated ({readm_rate:.1%}). Efficiency audit recommended."
                    )
                })

        # 3. Doctor Performance Deviation (Explicit Logic)
        doc = resolve_column(df, "doctor")
        readm = resolve_column(df, "readmitted")
        
        if doc and readm and pd.api.types.is_numeric_dtype(df[readm]):
            # Only run if we have enough doctors
            if df[doc].nunique() > 2:
                overall = df[readm].mean()
                grp = df.groupby(doc)[readm].mean()
                worst_doc = grp.idxmax()
                worst_val = grp.max()
                
                # If worst doctor is > 20% worse than average
                if worst_val > (overall * 1.20) and overall > 0:
                     insights.append({
                        "level": "RISK",
                        "title": "Provider Performance Variance",
                        "so_what": (
                            f"Dr. {worst_doc} has a readmission rate of {worst_val:.1%}, "
                            f"significantly higher than the facility average ({overall:.1%})."
                        )
                    })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        
        readm = kpis.get("readmission_rate")
        los = kpis.get("avg_length_of_stay")
        
        if readm is not None and readm > 0.15:
            recs.append({
                "action": "Review discharge planning and post-acute care follow-ups",
                "priority": "HIGH",
                "timeline": "Immediate",
            })
            
        if los is not None and los > 8.0:
             recs.append({
                "action": "Audit patient flow to identify bottlenecks in care delivery",
                "priority": "MEDIUM",
                "timeline": "Next Month",
            })

        if not recs:
            recs.append({
                "action": "Continue routine clinical monitoring",
                "priority": "LOW",
                "timeline": "Ongoing",
            })

        return recs


# =====================================================
# DOMAIN DETECTOR (SMART / DOMINANCE BASED)
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
        
        # 🔑 HEALTHCARE DOMINANCE RULE
        # Strong terms that almost guarantee healthcare data
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
