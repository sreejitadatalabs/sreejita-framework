import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# DATA NORMALIZATION (CRITICAL FOR REAL DATASETS)
# =====================================================

def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No, Y/N, True/False columns to 1/0.
    Works safely for mixed datasets.
    """
    for col in df.columns:
        if df[col].dtype == object:
            values = set(
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )

            if values and values.issubset(
                {"yes", "no", "y", "n", "true", "false"}
            ):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({
                        "yes": 1, "y": 1, "true": 1,
                        "no": 0, "n": 0, "false": 0,
                    })
                )
    return df


# =====================================================
# CAPABILITY DETECTION
# =====================================================

def _detect_capabilities(df: pd.DataFrame) -> Dict[str, bool]:
    return {
        "has_patient": resolve_column(df, "patient_id") is not None,
        "has_clinical": any(
            resolve_column(df, k) is not None
            for k in ["length_of_stay", "readmitted", "mortality"]
        ),
        "has_financials": resolve_column(df, "billing_amount") is not None,
        "has_provider": resolve_column(df, "doctor") is not None,
        "has_diagnosis": resolve_column(df, "diagnosis") is not None,
        "has_insurance": resolve_column(df, "insurance") is not None,
    }


# =====================================================
# HEALTHCARE DOMAIN ENGINE
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Generalized healthcare analytics (clinical + financial + operational)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return resolve_column(df, "patient_id") is not None

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1️⃣ Normalize real-world categorical data
        df = _normalize_binary_columns(df)

        # 2️⃣ Detect dataset capabilities
        self.capabilities = _detect_capabilities(df)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        pid = resolve_column(df, "patient_id")

        # ---- Clinical KPIs ----
        if self.capabilities["has_clinical"]:
            los = resolve_column(df, "length_of_stay")
            if los and pd.api.types.is_numeric_dtype(df[los]):
                kpis["avg_length_of_stay"] = df[los].mean()

            readm = resolve_column(df, "readmitted")
            if readm and pd.api.types.is_numeric_dtype(df[readm]):
                kpis["readmission_rate"] = df[readm].mean()

            mort = resolve_column(df, "mortality")
            if mort and pd.api.types.is_numeric_dtype(df[mort]):
                kpis["mortality_rate"] = df[mort].mean()

        # ---- Financial KPIs ----
        if self.capabilities["has_financials"]:
            bill = resolve_column(df, "billing_amount")
            if bill and pd.api.types.is_numeric_dtype(df[bill]):
                kpis["total_billing"] = df[bill].sum()

                if pid:
                    kpis["avg_billing_per_patient"] = (
                        df.groupby(pid)[bill].sum().mean()
                    )

        # ---- Volume KPI ----
        if pid:
            kpis["patient_volume"] = df[pid].nunique()

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:
        import matplotlib.pyplot as plt

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- LOS Distribution ----
        los = resolve_column(df, "length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            path = output_dir / "length_of_stay.png"
            df[los].dropna().hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({
                "path": path,
                "caption": "Distribution of patient length of stay"
            })

        # ---- Billing by Insurance ----
        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins and pd.api.types.is_numeric_dtype(df[bill]):
            path = output_dir / "billing_by_insurance.png"
            df.groupby(ins)[bill].sum().sort_values(
                ascending=False
            ).plot(kind="bar")
            plt.title("Billing by Insurance Provider")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({
                "path": path,
                "caption": "Total billing amount by insurance provider"
            })

        return visuals

    # ---------------- INSIGHTS ----------------

    def generate_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

    # ---------- Helper ----------
        def pct_diff(val, ref):
            if ref == 0:
                return 0
            return (val - ref) / ref

    # ---------- Worst Performing Doctor (Readmission) ----------
        doc = resolve_column(df, "doctor")
        readm = resolve_column(df, "readmitted")

        if doc and readm and pd.api.types.is_numeric_dtype(df[readm]):
            overall_rate = df[readm].mean()
            grp = df.groupby(doc)[readm].mean()

            worst = grp.loc[grp.idxmax()]
            diff = pct_diff(worst, overall_rate)

            if diff > 0.10:  # 10% worse than avg
                insights.append({
                    "level": "RISK",
                    "title": f"Worst Performing Doctor: {grp.idxmax()}",
                    "so_what": (
                        f"Readmission rate ({worst:.1%}) is "
                        f"{diff:.0%} higher than the overall average ({overall_rate:.1%})."
                    )
                })

    # ---------- Highest Cost Diagnosis ----------
        diag = resolve_column(df, "diagnosis")
        bill = resolve_column(df, "billing_amount")

        if diag and bill and pd.api.types.is_numeric_dtype(df[bill]):
            overall_bill = df[bill].mean()
            grp = df.groupby(diag)[bill].mean()

            worst_diag = grp.idxmax()
            worst_val = grp.loc[worst_diag]
            diff = pct_diff(worst_val, overall_bill)

            if diff > 0.25:  # 25% higher cost
                insights.append({
                    "level": "WARNING",
                    "title": f"High-Cost Diagnosis: {worst_diag}",
                    "so_what": (
                        f"Average billing for this diagnosis is "
                        f"{diff:.0%} higher than the overall average."
                    )
                })

    # ---------- Most Expensive Branch ----------
    branch = resolve_column(df, "hospital_branch")
    if branch and bill and pd.api.types.is_numeric_dtype(df[bill]):
        grp = df.groupby(branch)[bill].mean()
        worst_branch = grp.idxmax()
        worst_val = grp.loc[worst_branch]
        avg_val = grp.mean()

        diff = pct_diff(worst_val, avg_val)

        if diff > 0.20:
            insights.append({
                "level": "WARNING",
                "title": f"Most Expensive Branch: {worst_branch}",
                "so_what": (
                    f"Average billing is {diff:.0%} higher than "
                    f"the branch average."
                )
            })

    # ---------- Fallback (Should Almost Never Trigger Now) ----------
    if not insights:
        insights.append({
            "level": "INFO",
            "title": "No Extreme Outliers Detected",
            "so_what": (
                "All key metrics fall within normal variance ranges, "
                "with no segment deviating materially from the mean."
            )
        })

    return insights

    # ---------------- RECOMMENDATIONS ----------------

def generate_recommendations(
    self, df: pd.DataFrame, kpis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []

    insights = self.generate_insights(df, kpis)

    for i in insights:
        if i["level"] == "RISK":
            recs.append({
                "action": f"Investigate and remediate: {i['title']}",
                "priority": "HIGH",
                "timeline": "2–4 weeks",
            })
        elif i["level"] == "WARNING":
            recs.append({
                "action": f"Review cost and efficiency drivers for: {i['title']}",
                "priority": "MEDIUM",
                "timeline": "4–6 weeks",
            })

    if not recs:
        recs.append({
            "action": "Continue routine monitoring across all clinical and financial dimensions",
            "priority": "LOW",
            "timeline": "Ongoing",
        })

    return recs


# =====================================================
# DOMAIN DETECTOR (TEST + ROUTER SAFE)
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_COLUMNS: Set[str] = {
        "patient", "patient_id", "pid",
        "los", "length_of_stay",
        "readmitted", "mortality",
        "billing", "insurance",
        "doctor", "diagnosis",
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.HEALTHCARE_COLUMNS)
        confidence = min(len(matches) / 4, 1.0)

        return DomainDetectionResult(
            domain="healthcare",
            confidence=confidence,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="healthcare",
        domain_cls=HealthcareDomain,
        detector_cls=HealthcareDomainDetector,
    )
