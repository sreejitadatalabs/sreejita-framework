"""
Customer Domain Module (v2.x FINAL)

- Ranked KPI fallback system (top 4 guaranteed if data allows)
- KPI-driven visuals (rank-aligned)
- Defensive execution (schema-safe)
- Hybrid/domain-neutral reporting compatible
"""

from typing import Dict, Any, List, Set
from pathlib import Path
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# KPI RANKING PLAN (AUTHORITATIVE CONTRACT)
# =====================================================

CUSTOMER_KPI_PLAN = [
    {"name": "customer_count", "rank": 1, "columns": ["customer_id"]},
    {"name": "avg_lifetime_value", "rank": 2, "columns": ["lifetime_value"]},
    {"name": "churn_rate", "rank": 3, "columns": ["churn"]},
    {"name": "avg_recency", "rank": 4, "columns": ["recency"]},
    {"name": "avg_frequency", "rank": 5, "columns": ["frequency"]},
    {"name": "avg_monetary", "rank": 6, "columns": ["monetary"]},
    {"name": "engagement_score", "rank": 7, "columns": ["engagement"]},
    {"name": "segment_distribution", "rank": 8, "columns": ["segment"]},
    {"name": "channel_mix", "rank": 9, "columns": ["channel"]},
    {"name": "device_mix", "rank": 10, "columns": ["device"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(CUSTOMER_KPI_PLAN, key=lambda x: x["rank"]):
        if all(col in df.columns for col in item["columns"]):
            selected.append(item["name"])
        if len(selected) >= max_kpis:
            break
    return selected


# =====================================================
# KPI COMPUTATION (DEFENSIVE)
# =====================================================

def _compute_kpi(name: str, df: pd.DataFrame):
    try:
        if name == "customer_count":
            return int(df["customer_id"].nunique())

        if name == "avg_lifetime_value":
            return float(df["lifetime_value"].mean())

        if name == "churn_rate":
            return float(df["churn"].mean())

        if name == "avg_recency":
            return float(df["recency"].mean())

        if name == "avg_frequency":
            return float(df["frequency"].mean())

        if name == "avg_monetary":
            return float(df["monetary"].mean())

        if name == "engagement_score":
            return float(df["engagement"].mean())

        if name == "segment_distribution":
            return df["segment"].value_counts(normalize=True).iloc[0]

        if name == "channel_mix":
            return df["channel"].value_counts(normalize=True).iloc[0]

        if name == "device_mix":
            return df["device"].value_counts(normalize=True).iloc[0]

    except Exception:
        return None

    return None


# =====================================================
# VISUALS (ONLY FOR SELECTED KPIs)
# =====================================================

def _generate_visuals(df: pd.DataFrame, selected_kpis: List[str], out_dir: Path):
    """
    Visuals are KPI-driven and rank-aligned.
    Only a subset is implemented intentionally.
    """
    visuals = []
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    if "avg_lifetime_value" in selected_kpis:
        path = out_dir / "ltv_distribution.png"
        df["lifetime_value"].hist()
        plt.title("Customer Lifetime Value Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of customer lifetime value"
        })

    if "churn_rate" in selected_kpis:
        path = out_dir / "churn_rate.png"
        df["churn"].value_counts().plot(kind="bar")
        plt.title("Churn Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Customer churn overview"
        })

    if "segment_distribution" in selected_kpis:
        path = out_dir / "segment_distribution.png"
        df["segment"].value_counts().plot(kind="bar")
        plt.title("Customer Segments")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of customer segments"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class CustomerDomain(BaseDomain):
    name = "customer"
    description = "Customer analytics with ranked KPI fallback"
    required_columns = ["customer_id"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "customer_id" in df.columns

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

        if "churn_rate" in kpis and kpis["churn_rate"] > 0.25:
            insights.append({
                "level": "RISK",
                "title": "High Customer Churn",
                "value": f"{kpis['churn_rate']:.1%}",
                "so_what": "High churn directly reduces lifetime value and revenue stability."
            })

        if "avg_lifetime_value" in kpis and kpis["avg_lifetime_value"] < 500:
            insights.append({
                "level": "WARNING",
                "title": "Low Customer Lifetime Value",
                "value": f"${kpis['avg_lifetime_value']:.0f}",
                "so_what": "Low LTV limits sustainable growth."
            })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "churn_rate" in kpis and kpis["churn_rate"] > 0.25:
            recs.append({
                "action": "Launch retention and loyalty programs",
                "expected_impact": "10–20% churn reduction",
                "timeline": "3–6 weeks"
            })

        if "avg_lifetime_value" in kpis and kpis["avg_lifetime_value"] < 500:
            recs.append({
                "action": "Improve upsell and cross-sell strategies",
                "expected_impact": "+15–25% LTV increase",
                "timeline": "4–8 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class CustomerDomainDetector(BaseDomainDetector):
    domain_name = "customer"

    CUSTOMER_COLUMNS: Set[str] = {
        "customer_id", "customer_name", "email", "phone",
        "segment", "recency", "frequency", "monetary",
        "lifetime_value", "churn", "engagement",
        "channel", "device"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("customer", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.CUSTOMER_COLUMNS)

        score = min((len(matches) / len(self.CUSTOMER_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="customer",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="customer",
        domain_cls=CustomerDomain,
        detector_cls=CustomerDomainDetector,
    )
