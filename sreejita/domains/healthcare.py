"""
Healthcare Domain Module (v2.x FINAL)

- Ranked KPI fallback system (top 4 always attempted)
- KPI-driven visuals (rank-aligned)
- Defensive execution (schema-safe)
- Domain-neutral reporting compatible
"""

from typing import Dict, Any, List, Set
from pathlib import Path
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# KPI RANKING PLAN (AUTHORITATIVE CONTRACT)
# =====================================================

HEALTHCARE_KPI_PLAN = [
    {"name": "readmission_rate", "rank": 1, "columns": ["readmitted"]},
    {"name": "avg_length_of_stay", "rank": 2, "columns": ["length_of_stay"]},
    {"name": "avg_outcome_score", "rank": 3, "columns": ["outcome_score"]},
    {"name": "mortality_rate", "rank": 4, "columns": ["mortality"]},
    {"name": "treatment_cost", "rank": 5, "columns": ["treatment_cost"]},
    {"name": "patient_volume", "rank": 6, "columns": ["patient_id"]},
    {"name": "icu_utilization", "rank": 7, "columns": ["icu_days"]},
    {"name": "avg_age", "rank": 8, "columns": ["age"]},
    {"name": "gender_mix", "rank": 9, "columns": ["gender"]},
    {"name": "insurance_mix", "rank": 10, "columns": ["insurance_provider"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(HEALTHCARE_KPI_PLAN, key=lambda x: x["rank"]):
        if all(col in df.columns for col in item["columns"]):
            selected.append(item["name"])
        if len(selected) >= max_kpis:
            break
    return selected


def _safe_div(n, d):
    return float(n / d) if d and d != 0 else None


# =====================================================
# KPI COMPUTATION (DEFENSIVE)
# =====================================================

def _compute_kpi(name: str, df: pd.DataFrame):
    try:
        if name == "readmission_rate":
            return float(df["readmitted"].mean())

        if name == "avg_length_of_stay":
            return float(df["length_of_stay"].mean())

        if name == "avg_outcome_score":
            return float(df["outcome_score"].mean())

        if name == "mortality_rate":
            return float(df["mortality"].mean())

        if name == "treatment_cost":
            return float(df["treatment_cost"].mean())

        if name == "patient_volume":
            return int(df["patient_id"].nunique())

        if name == "icu_utilization":
            return float(df["icu_days"].mean())

        if name == "avg_age":
            return float(df["age"].mean())

        if name == "gender_mix":
            return df["gender"].value_counts(normalize=True).iloc[0]

        if name == "insurance_mix":
            return df["insurance_provider"].value_counts(normalize=True).iloc[0]

    except Exception:
        return None

    return None


# =====================================================
# VISUALS (ONLY FOR SELECTED KPIs)
# =====================================================

def _generate_visuals(df: pd.DataFrame, selected_kpis: List[str], out_dir: Path):
    """
    Visuals are KPI-driven and rank-aligned.
    Only a few visuals are implemented intentionally.
    """
    visuals = []
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    if "avg_length_of_stay" in selected_kpis:
        path = out_dir / "length_of_stay.png"
        df["length_of_stay"].hist()
        plt.title("Length of Stay Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of patient length of stay"
        })

    if "readmission_rate" in selected_kpis:
        path = out_dir / "readmission_rate.png"
        df["readmitted"].value_counts().plot(kind="bar")
        plt.title("Readmission Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Readmission frequency overview"
        })

    if "mortality_rate" in selected_kpis:
        path = out_dir / "mortality_rate.png"
        df["mortality"].value_counts().plot(kind="bar")
        plt.title("Mortality Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Mortality outcome distribution"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Healthcare analytics with ranked KPI fallback"
    required_columns = ["patient_id"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "patient_id" in df.columns or "outcome_score" in df.columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        selected = _select_ranked_kpis(df)
        kpis = {}

        for name in selected:
            val = _compute_kpi(name, df)
            if val is not None:
                kpis[name] = val

        return kpis

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        if "readmission_rate" in kpis and kpis["readmission_rate"] > 0.2:
            insights.append({
                "level": "RISK",
                "title": "High Readmission Rate",
                "value": f"{kpis['readmission_rate']:.1%}",
                "so_what": "High readmissions indicate care quality or discharge gaps."
            })

        if "avg_length_of_stay" in kpis and kpis["avg_length_of_stay"] > 7:
            insights.append({
                "level": "WARNING",
                "title": "Extended Length of Stay",
                "value": f"{kpis['avg_length_of_stay']:.1f} days",
                "so_what": "Long stays increase costs and reduce capacity."
            })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "readmission_rate" in kpis and kpis["readmission_rate"] > 0.2:
            recs.append({
                "action": "Strengthen discharge planning and follow-ups",
                "expected_impact": "5–10% reduction in readmissions",
                "timeline": "4–6 weeks"
            })

        if "avg_length_of_stay" in kpis and kpis["avg_length_of_stay"] > 7:
            recs.append({
                "action": "Optimize inpatient workflow and care coordination",
                "expected_impact": "1–2 day reduction in average stay",
                "timeline": "6–8 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_COLUMNS: Set[str] = {
        "patient_id", "admission", "discharge",
        "length_of_stay", "readmitted", "mortality",
        "outcome_score", "treatment_cost", "doctor",
        "hospital", "insurance_provider"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("healthcare", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.HEALTHCARE_COLUMNS)

        score = min((len(matches) / len(self.HEALTHCARE_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="healthcare",
            confidence=score,
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
