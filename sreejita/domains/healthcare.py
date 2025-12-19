import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# CAPABILITY DETECTION
# =====================================================

def _detect_capabilities(df):
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


class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Generalized healthcare analytics"

    def validate_data(self, df: pd.DataFrame) -> bool:
        return resolve_column(df, "patient_id") is not None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 🔥 THIS IS CRITICAL
        self.capabilities = _detect_capabilities(df)
        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}

        if self.capabilities.get("has_clinical"):
            los = resolve_column(df, "length_of_stay")
            if los:
                kpis["avg_length_of_stay"] = df[los].mean()

            readm = resolve_column(df, "readmitted")
            if readm:
                kpis["readmission_rate"] = df[readm].mean()

        if self.capabilities.get("has_financials"):
            bill = resolve_column(df, "billing_amount")
            pid = resolve_column(df, "patient_id")
            if bill:
                kpis["total_billing"] = df[bill].sum()
                if pid:
                    kpis["avg_billing_per_patient"] = (
                        df.groupby(pid)[bill].sum().mean()
                    )

        pid = resolve_column(df, "patient_id")
        if pid:
            kpis["patient_volume"] = df[pid].nunique()

        return kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        import matplotlib.pyplot as plt

        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        los = resolve_column(df, "length_of_stay")
        if los:
            path = output_dir / "length_of_stay.png"
            df[los].hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({
                "path": path,
                "caption": "Distribution of patient length of stay"
            })

        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins:
            path = output_dir / "billing_by_insurance.png"
            df.groupby(ins)[bill].sum().plot(kind="bar")
            plt.title("Billing by Insurance Provider")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({
                "path": path,
                "caption": "Billing by insurance provider"
            })

        return visuals

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        if "readmission_rate" in kpis and kpis["readmission_rate"] > 0.2:
            insights.append({
                "level": "RISK",
                "title": "High Readmission Rate",
                "so_what": "Potential discharge or follow-up gaps detected."
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Operational Performance Stable",
                "so_what": "No critical clinical or financial risks detected."
            })

        return insights


class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    HEALTHCARE_COLUMNS: Set[str] = {
        "patient", "patient_id", "los", "readmitted",
        "billing", "insurance", "doctor", "diagnosis"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {c.lower() for c in df.columns}
        matches = cols.intersection(self.HEALTHCARE_COLUMNS)
        confidence = min(len(matches) / 4, 1.0)

        return DomainDetectionResult(
            domain="healthcare",
            confidence=confidence,
            signals={"matched_columns": list(matches)}
        )


def register(registry):
    registry.register("healthcare", HealthcareDomain, HealthcareDomainDetector)
