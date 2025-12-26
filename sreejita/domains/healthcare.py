import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import matplotlib
matplotlib.use("Agg")
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
# HEALTHCARE DOMAIN (v3.0 - FULL AUTHORITY)
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
        bill = resolve_column(df, "billing_amount") or resolve_column(df, "cost")
        pid = resolve_column(df, "patient_id")

        # 1. Operational: Length of Stay
        if los and pd.api.types.is_numeric_dtype(df[los]) and df[los].notna().any():
            kpis["avg_length_of_stay"] = df[los].mean()
            kpis["target_avg_los"] = 5.0 # Benchmark target

        # 2. Clinical: Readmission Rate
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()
            kpis["target_readmission_rate"] = 0.10 # Benchmark target

        # 3. Financial: Billing/Cost
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            # Map to standardized key for insights
            kpis["avg_treatment_cost"] = df[bill].mean()
            
            if pid:
                kpis["avg_billing_per_patient"] = df.groupby(pid)[bill].sum().mean()

        # 4. Volume
        if pid:
            kpis["patient_volume"] = df[pid].nunique()
        else:
            kpis["patient_volume"] = len(df)

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        import matplotlib
        matplotlib.use("Agg")  # REQUIRED for headless environments

        visuals: List[Dict[str, Any]] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        def human_fmt(x, _):
            if x >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if x >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))
    
        # -------- Visual 1: Length of Stay Distribution --------
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and df[los].notna().any() and pd.api.types.is_numeric_dtype(df[los]):
            p = output_dir / "length_of_stay.png"
            plt.figure(figsize=(7, 4))
            df[los].dropna().hist(bins=15, color="#1f77b4", edgecolor="white")
            plt.title("Length of Stay Distribution")
            plt.xlabel("Days")
            plt.tight_layout()
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            visuals.append({
                "path": str(p),
                "caption": "Patient stay duration distribution",
            })
    
        # -------- Visual 2: Billing by Insurance --------
        bill = resolve_column(df, "billing_amount") or resolve_column(df, "cost")
        ins = resolve_column(df, "insurance")
        if bill and ins and pd.api.types.is_numeric_dtype(df[bill]):
            p = output_dir / "billing_by_insurance.png"
            plt.figure(figsize=(7, 4))
            ax = (
                df.groupby(ins)[bill]
                .sum()
                .sort_values(ascending=False)
                .head(7)
                .plot(kind="bar", color="#2ca02c")
            )
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.title("Billing by Insurance Provider")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            visuals.append({
                "path": str(p),
                "caption": "Revenue concentration by payer",
            })
    
        # -------- Visual 3: Avg Cost by Condition --------
        diag = resolve_column(df, "diagnosis") or resolve_column(df, "medical_condition")
        if diag and bill and pd.api.types.is_numeric_dtype(df[bill]):
            p = output_dir / "cost_by_condition.png"
            plt.figure(figsize=(7, 4))
            (
                df.groupby(diag)[bill]
                .mean()
                .sort_values(ascending=False)
                .head(7)
                .plot(kind="barh", color="#d62728")
            )
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.title("Avg Cost by Condition (Top 7)")
            plt.tight_layout()
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            visuals.append({
                "path": str(p),
                "caption": "Average treatment cost by condition",
            })
    
        return visuals

    # ---------------- ATOMIC INSIGHTS (WITH DOMINANCE RULE) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        # 1. Text Scanning (Categorical Risks)
        insights.extend(_scan_categorical_risks(df))

        # === STEP 1: Composite FIRST (Authority Layer) ===
        composite_insights = []
        if len(df) > 30:
            composite_insights = self.generate_composite_insights(df, kpis)

        dominant_titles = {
            i["title"] for i in composite_insights
            if i["level"] in {"RISK", "WARNING"}
        }

        # === STEP 2: Suppression Rules ===
        # If "Operational Strain" is found, suppress atomic "High LOS" and "Readmission"
        suppress_los = "Operational Strain Detected" in dominant_titles
        suppress_readmission = "Operational Strain Detected" in dominant_titles
        
        # If "Financial Toxicity" is found, suppress atomic "High Cost"
        suppress_cost = "Financial Toxicity Risk" in dominant_titles

        # === STEP 3: Guarded Atomic Insights ===
        los = kpis.get("avg_length_of_stay")
        readm = kpis.get("readmission_rate")
        cost = kpis.get("avg_treatment_cost")

        if los is not None and not suppress_los:
            if los > 7:
                insights.append({
                    "level": "WARNING",
                    "title": "Extended Length of Stay",
                    "so_what": f"Average LOS is {los:.1f} days, increasing bed occupancy pressure."
                })

        if readm is not None and not suppress_readmission:
            if readm > 0.15:
                insights.append({
                    "level": "RISK",
                    "title": "High Readmission Rate",
                    "so_what": f"Readmission rate is {readm:.1%}, indicating post-discharge gaps."
                })
            elif readm > 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Elevated Readmissions",
                    "so_what": f"Rate is {readm:.1%}."
                })

        if cost is not None and not suppress_cost:
            # Simple threshold for demo; ideally benchmarked
            if cost > 50000: 
                insights.append({
                    "level": "WARNING",
                    "title": "High Treatment Cost",
                    "so_what": f"Average treatment cost is elevated at {cost:,.0f}."
                })

        # === STEP 4: Composite Insights LAST (Authority Wins) ===
        insights += composite_insights

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Healthcare Operations Stable",
                "so_what": "Clinical, operational, and financial indicators are within acceptable limits."
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

        los = kpis.get("avg_length_of_stay")
        readm = kpis.get("readmission_rate")
        cost = kpis.get("avg_treatment_cost")

        # 1. Operational Strain (High LOS + High Readmission)
        if los is not None and readm is not None:
            if los > 7 and readm > 0.15:
                insights.append({
                    "level": "RISK",
                    "title": "Operational Strain Detected",
                    "so_what": (
                        f"Extended LOS ({los:.1f} days) combined with high readmissions "
                        f"({readm:.1%}) suggests discharge workflow and capacity strain."
                    )
                })

        # 2. Financial Toxicity (High Cost + Poor Outcome)
        if cost is not None and readm is not None:
            if cost > 50000 and readm > 0.15:
                insights.append({
                    "level": "WARNING",
                    "title": "Financial Toxicity Risk",
                    "so_what": (
                        f"High treatment costs ({cost:,.0f}) with poor outcomes "
                        f"(readmissions {readm:.1%}) indicate inefficiencies."
                    )
                })
                
        # 3. Provider Variance (Doctor Specific)
        doc = resolve_column(df, "doctor")
        readm_col = resolve_column(df, "readmitted")
        
        if doc and readm_col and pd.api.types.is_numeric_dtype(df[readm_col]):
            if df[doc].nunique() > 2:
                overall = df[readm_col].mean()
                grp = df.groupby(doc)[readm_col].mean()
                worst_doc = grp.idxmax()
                worst_val = grp.max()
                
                if worst_val > (overall * 1.30) and overall > 0:
                     insights.append({
                        "level": "RISK",
                        "title": "Provider Performance Variance",
                        "so_what": (
                            f"Dr. {worst_doc} has a readmission rate of {worst_val:.1%}, "
                            f"significantly higher than the facility average ({overall:.1%})."
                        )
                    })

        return insights

    # ---------------- RECOMMENDATIONS (AUTHORITY BASED) ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []

        # 1. Check Composite Context
        composite = []
        if len(df) > 30:
            composite = self.generate_composite_insights(df, kpis)

        titles = [i["title"] for i in composite]

        # === AUTHORITY RULES (Mandatory Actions) ===
        if "Operational Strain Detected" in titles:
            return [{
                "action": "Audit discharge planning, bed management, and care coordination workflows",
                "priority": "HIGH",
                "timeline": "Immediate"
            }]

        if "Financial Toxicity Risk" in titles:
            return [{
                "action": "Review treatment protocols and cost-efficiency for high-risk cases",
                "priority": "HIGH",
                "timeline": "This Month"
            }]

        if "Provider Performance Variance" in titles:
             return [{
                "action": "Initiate peer review and clinical standardization for outlier providers",
                "priority": "HIGH",
                "timeline": "This Quarter"
            }]

        # === FALLBACK (Atomic Recs) ===
        if kpis.get("readmission_rate", 0) > 0.15:
            recs.append({
                "action": "Strengthen post-discharge follow-up programs",
                "priority": "MEDIUM",
                "timeline": "This Quarter"
            })
            
        if kpis.get("avg_length_of_stay", 0) > 8.0:
            recs.append({
                "action": "Investigate patient flow bottlenecks",
                "priority": "MEDIUM",
                "timeline": "Next Month"
            })

        if not recs:
            recs.append({
                "action": "Continue monitoring healthcare performance indicators",
                "priority": "LOW",
                "timeline": "Ongoing"
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
