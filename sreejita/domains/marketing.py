"""
Marketing Domain Module (v2.x FINAL)

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

MARKETING_KPI_PLAN = [
    {"name": "total_ad_spend", "rank": 1, "columns": ["ad_spend"]},
    {"name": "total_impressions", "rank": 2, "columns": ["impressions"]},
    {"name": "total_clicks", "rank": 3, "columns": ["clicks"]},
    {"name": "ctr", "rank": 4, "columns": ["clicks", "impressions"]},
    {"name": "conversion_rate", "rank": 5, "columns": ["conversions", "clicks"]},
    {"name": "cpc", "rank": 6, "columns": ["ad_spend", "clicks"]},
    {"name": "roas", "rank": 7, "columns": ["revenue", "ad_spend"]},
    {"name": "campaign_count", "rank": 8, "columns": ["campaign_id"]},
    {"name": "channel_mix", "rank": 9, "columns": ["channel"]},
    {"name": "region_mix", "rank": 10, "columns": ["region"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(MARKETING_KPI_PLAN, key=lambda x: x["rank"]):
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
        if name == "total_ad_spend":
            return float(df["ad_spend"].sum())

        if name == "total_impressions":
            return int(df["impressions"].sum())

        if name == "total_clicks":
            return int(df["clicks"].sum())

        if name == "ctr":
            return _safe_div(df["clicks"].sum(), df["impressions"].sum())

        if name == "conversion_rate":
            return _safe_div(df["conversions"].sum(), df["clicks"].sum())

        if name == "cpc":
            return _safe_div(df["ad_spend"].sum(), df["clicks"].sum())

        if name == "roas":
            return _safe_div(df["revenue"].sum(), df["ad_spend"].sum())

        if name == "campaign_count":
            return int(df["campaign_id"].nunique())

        if name == "channel_mix":
            return df["channel"].value_counts(normalize=True).iloc[0]

        if name == "region_mix":
            return df["region"].value_counts(normalize=True).iloc[0]

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

    if "total_ad_spend" in selected_kpis:
        path = out_dir / "ad_spend.png"
        df["ad_spend"].plot(kind="hist")
        plt.title("Ad Spend Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of advertising spend"
        })

    if "ctr" in selected_kpis:
        path = out_dir / "ctr.png"
        (_safe_div(df["clicks"], df["impressions"])).plot(kind="hist")
        plt.title("Click-Through Rate Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of click-through rates"
        })

    if "channel_mix" in selected_kpis:
        path = out_dir / "channel_mix.png"
        df["channel"].value_counts().plot(kind="bar")
        plt.title("Marketing Channel Mix")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Channel-wise distribution of campaigns"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class MarketingDomain(BaseDomain):
    name = "marketing"
    description = "Marketing analytics with ranked KPI fallback"
    required_columns = ["ad_spend"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "ad_spend" in df.columns or "campaign_id" in df.columns

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

        if "ctr" in kpis and kpis["ctr"] < 0.01:
            insights.append({
                "level": "WARNING",
                "title": "Low Click-Through Rate",
                "value": f"{kpis['ctr']:.2%}",
                "so_what": "Low CTR indicates weak creative or poor audience targeting."
            })

        if "roas" in kpis and kpis["roas"] < 1.0:
            insights.append({
                "level": "RISK",
                "title": "Negative ROAS",
                "value": f"{kpis['roas']:.2f}",
                "so_what": "Campaign spend is not generating sufficient revenue."
            })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "ctr" in kpis and kpis["ctr"] < 0.01:
            recs.append({
                "action": "Refresh creatives and refine audience targeting",
                "expected_impact": "10–20% CTR improvement",
                "timeline": "2–3 weeks"
            })

        if "roas" in kpis and kpis["roas"] < 1.0:
            recs.append({
                "action": "Reallocate budget to high-performing campaigns",
                "expected_impact": "Positive ROAS within 4–6 weeks",
                "timeline": "4–6 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class MarketingDomainDetector(BaseDomainDetector):
    domain_name = "marketing"

    MARKETING_COLUMNS: Set[str] = {
        "campaign", "campaign_id", "impressions", "clicks",
        "ctr", "ad_spend", "cpc", "roas", "conversions",
        "channel", "audience", "creative", "region", "revenue"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("marketing", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.MARKETING_COLUMNS)

        score = min((len(matches) / len(self.MARKETING_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="marketing",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="marketing",
        domain_cls=MarketingDomain,
        detector_cls=MarketingDomainDetector,
    )
