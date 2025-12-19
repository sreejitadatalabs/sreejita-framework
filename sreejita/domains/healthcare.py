import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# DATA NORMALIZATION
# =====================================================

def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No, Y/N, True/False columns to 1/0.
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
            if values and values.issubset({"yes", "no", "y", "n", "true", "false"}):
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
# DATE INTELLIGENCE (DERIVED LOS)
# =====================================================

def _derive_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive length_of_stay if missing but admission & discharge dates exist.
    """
    if resolve_column(df, "length_of_stay"):
        return df

    admit = resolve_column(df, "admission_date")
    discharge = resolve_column(df, "discharge_date")

    if admit and discharge:
        try:
            df[admit] = pd.to_datetime(df[admit], errors="coerce")
            df[discharge] = pd.to_datetime(df[discharge], errors="coerce")
            df["derived_length_of_stay"] = (
                df[discharge] - df[admit]
            ).dt.days
        except Exception:
            pass

    return df


# =====================================================
# CAPABILITY DETECTION
# =====================================================

def _detect_capabilities(df: pd.DataFrame) -> Dict[str, bool]:
    return {
        "has_patient": resolve_column(df, "patient_id") is not None,
        "has_clinical": any(
            resolve_column(df, k) is not None
            for k in ["length_of_stay", "derived_length_of_stay", "readmitted", "mortality"]
        ),
        "has_financials": resolve_column(df, "billing_amount") is not None,
        "has_provider": resolve_column(df, "doctor") is not None,
        "has_diagnosis": resolve_column(df, "diagnosis") is not None,
        "has_insurance": resolve_column(df, "insurance") is not None,
    }


# =====================================================
# DYNAMIC CATEGORICAL RISK SCANNER
# =====================================================

NEGATIVE_KEYWORDS = {
    "abnormal", "failed", "deceased", "critical",
    "positive", "poor", "expired", "high"
}

OUTCOME_COLUMN_HINTS = {
    "result", "status", "outcome", "test", "finding"
}


def _scan_categorical_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    for col in df.columns:
        col_l = col.lower()
        if not any(h in col_l for h in OUTCOME_COLUMN_HINTS):
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
                "title": f"High Rate of Negative Outcomes in {col}",
                "so_what": (
                    f"{rate:.0%} of records contain negative values "
                    f"(e.g., {', '.join(negatives.unique()[:3])})."
                )
            })

    return insights


# =====================================================
# HEALTHCARE DOMAIN
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Universal healthcare analytics (clinical + financial + operational)"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return resolve_column(df, "patient_id") is not None

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _normalize_binary_columns(df)
        df = _derive_length_of_stay(df)
        self.capabilities = _detect_capabilities(df)
        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        pid = resolve_column(df, "patient_id")

        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            kpis["avg_length_of_stay"] = df[los].mean()

        readm = resolve_column(df, "readmitted")
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()

        bill = resolve_column(df, "billing_amount")
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
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        def human_axis(x, _):
            if x >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if x >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            path = output_dir / "length_of_stay.png"
            df[los].dropna().hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({"path": path, "caption": "Length of stay distribution"})

        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins and pd.api.types.is_numeric_dtype(df[bill]):
            path = output_dir / "billing_by_insurance.png"
            ax = df.groupby(ins)[bill].sum().sort_values(ascending=False).plot(kind="bar")
            ax.yaxis.set_major_formatter(FuncFormatter(human_axis))
            plt.title("Billing by Insurance Provider")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({"path": path, "caption": "Billing by insurance provider"})

        return visuals

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        # Dynamic categorical risks
        insights.extend(_scan_categorical_risks(df))

        def pct_diff(v, r):
            return 0 if r == 0 else (v - r) / r

        # Worst doctor by readmission
        doc = resolve_column(df, "doctor")
        readm = resolve_column(df, "readmitted")
        if doc and readm and pd.api.types.is_numeric_dtype(df[readm]):
            overall = df[readm].mean()
            grp = df.groupby(doc)[readm].mean()
            name = grp.idxmax()
            val = grp.loc[name]
            if pct_diff(val, overall) > 0.10:
                insights.append({
                    "level": "RISK",
                    "title": f"Worst Performing Doctor: {name}",
                    "so_what": f"Readmission rate {val:.1%} vs avg {overall:.1%}."
                })

        # Ranking fallback (ALWAYS)
        if not insights:
            bill = resolve_column(df, "billing_amount")
            if bill:
                for key, label in [
                    ("insurance", "Insurance Provider"),
                    ("diagnosis", "Diagnosis"),
                    ("hospital_branch", "Hospital Branch"),
                ]:
                    col = resolve_column(df, key)
                    if col:
                        grp = df.groupby(col)[bill].sum()
                        top = grp.idxmax()
                        share = grp.loc[top] / grp.sum()
                        insights.append({
                            "level": "INFO",
                            "title": f"Highest Billing Concentration: {top}",
                            "so_what": (
                                f"The top {label.lower()} contributes "
                                f"{share:.0%} of total billing."
                            )
                        })
                        break

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
            elif i["level"] == "WARNING":
                recs.append({
                    "action": f"Review drivers for: {i['title']}",
                    "priority": "MEDIUM",
                    "timeline": "4–6 weeks",
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
            signals={"matched_columns": list(matches)},
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
