from typing import Dict, Any, List, Set
from pathlib import Path
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# COLUMN ALIAS MAP (DATASET INTELLIGENCE v2.x)
# =====================================================

COLUMN_ALIASES = {
    "readmitted": ["readmitted", "readmit", "re_admitted", "readdmitted"],
    "length_of_stay": ["length_of_stay", "los", "stay_length", "lengthofstay"],
    "outcome_score": ["outcome_score", "outcome", "clinical_score"],
    "mortality": ["mortality", "death", "is_dead"],
    "patient_id": ["patient_id", "patientid", "pid", "patient"],
    "age": ["age", "patient_age"],
}


def resolve_column(df: pd.DataFrame, aliases: List[str]):
    for col in aliases:
        if col in df.columns:
            return col
    return None


# =====================================================
# KPI PLAN (ORDER = BUSINESS PRIORITY)
# =====================================================

KPI_PLAN = [
    ("readmission_rate", "readmitted"),
    ("avg_length_of_stay", "length_of_stay"),
    ("avg_outcome_score", "outcome_score"),
    ("mortality_rate", "mortality"),
    ("patient_volume", "patient_id"),
    ("avg_age", "age"),
]


# =====================================================
# DOMAIN ENGINE
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Healthcare analytics with dataset-aware intelligence"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return any(
            resolve_column(df, COLUMN_ALIASES[key]) is not None
            for key in ["patient_id", "readmitted", "outcome_score"]
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}

        for kpi_name, canonical_col in KPI_PLAN:
            col = resolve_column(df, COLUMN_ALIASES.get(canonical_col, []))
            if not col:
                continue

            try:
                if kpi_name == "patient_volume":
                    kpis[kpi_name] = int(df[col].nunique())
                else:
                    kpis[kpi_name] = float(df[col].mean())
            except Exception:
                continue

            if len(kpis) >= 4:  # executive limit
                break

        return kpis

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        if "readmission_rate" in kpis and kpis["readmission_rate"] > 0.2:
            insights.append({
                "level": "RISK",
                "title": "High Readmission Rate",
                "so_what": "Indicates potential discharge or follow-up gaps."
            })

        if "avg_length_of_stay" in kpis and kpis["avg_length_of_stay"] > 7:
            insights.append({
                "level": "WARNING",
                "title": "Extended Length of Stay",
                "so_what": "Longer stays increase costs and reduce capacity."
            })

        if not insights and kpis:
            insights.append({
                "level": "INFO",
                "title": "Clinical Performance Stable",
                "so_what": "No major clinical risks detected in the current dataset."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "readmission_rate" in kpis and kpis["readmission_rate"] > 0.2:
            recs.append({
                "action": "Improve discharge planning and post-care follow-ups",
                "priority": "HIGH",
                "timeline": "4–6 weeks",
            })

        if not recs and kpis:
            recs.append({
                "action": "Maintain current clinical protocols and monitoring",
                "priority": "LOW",
                "timeline": "Ongoing",
            })

        return recs

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        import matplotlib.pyplot as plt

        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        los_col = resolve_column(df, COLUMN_ALIASES["length_of_stay"])
        if los_col:
            path = output_dir / "length_of_stay.png"
            df[los_col].dropna().hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({
                "path": path,
                "caption": "Distribution of patient length of stay"
            })

        readmit_col = resolve_column(df, COLUMN_ALIASES["readmitted"])
        if readmit_col:
            path = output_dir / "readmission.png"
            df[readmit_col].value_counts().plot(kind="bar")
            plt.title("Readmission Overview")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({
                "path": path,
                "caption": "Readmission frequency overview"
            })

        return visuals


# =====================================================
# DOMAIN DETECTOR (RESTORED — TEST SAFE)
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_COLUMNS: Set[str] = {
        "patient_id", "patientid", "pid",
        "readmitted", "readmit",
        "length_of_stay", "los",
        "outcome_score", "outcome",
        "mortality", "death",
        "age", "patient_age",
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("healthcare", 0.0, {"reason": "invalid_df"})

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
