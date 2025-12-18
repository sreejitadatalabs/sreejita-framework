"""
Ecommerce Domain Module (v2.x FINAL)

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

ECOMMERCE_KPI_PLAN = [
    {"name": "conversion_rate", "rank": 1, "columns": ["conversions", "sessions"]},
    {"name": "average_order_value", "rank": 2, "columns": ["order_value"]},
    {"name": "cart_abandonment_rate", "rank": 3, "columns": ["carts", "checkouts"]},
    {"name": "revenue", "rank": 4, "columns": ["revenue"]},
    {"name": "customer_acquisition_cost", "rank": 5, "columns": ["ad_spend", "customers"]},
    {"name": "lifetime_value", "rank": 6, "columns": ["lifetime_value"]},
    {"name": "return_rate", "rank": 7, "columns": ["returns", "orders"]},
    {"name": "payment_method_mix", "rank": 8, "columns": ["payment_method"]},
    {"name": "traffic_source_mix", "rank": 9, "columns": ["traffic_source"]},
    {"name": "device_mix", "rank": 10, "columns": ["device"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(ECOMMERCE_KPI_PLAN, key=lambda x: x["rank"]):
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
        if name == "conversion_rate":
            return _safe_div(df["conversions"].sum(), df["sessions"].sum())

        if name == "average_order_value":
            return float(df["order_value"].mean())

        if name == "cart_abandonment_rate":
            return _safe_div(df["carts"].sum() - df["checkouts"].sum(), df["carts"].sum())

        if name == "revenue":
            return float(df["revenue"].sum())

        if name == "customer_acquisition_cost":
            return _safe_div(df["ad_spend"].sum(), df["customers"].sum())

        if name == "lifetime_value":
            return float(df["lifetime_value"].mean())

        if name == "return_rate":
            return _safe_div(df["returns"].sum(), df["orders"].sum())

        if name == "payment_method_mix":
            return df["payment_method"].value_counts(normalize=True).iloc[0]

        if name == "traffic_source_mix":
            return df["traffic_source"].value_counts(normalize=True).iloc[0]

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

    if "conversion_rate" in selected_kpis:
        path = out_dir / "conversion_rate.png"
        (_safe_div(df["conversions"], df["sessions"])).hist()
        plt.title("Conversion Rate Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of conversion rates"
        })

    if "cart_abandonment_rate" in selected_kpis:
        path = out_dir / "cart_abandonment.png"
        (_safe_div(df["carts"] - df["checkouts"], df["carts"])).hist()
        plt.title("Cart Abandonment Rate Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Cart abandonment behavior overview"
        })

    if "traffic_source_mix" in selected_kpis:
        path = out_dir / "traffic_source_mix.png"
        df["traffic_source"].value_counts().plot(kind="bar")
        plt.title("Traffic Source Mix")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of traffic sources"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class EcommerceDomain(BaseDomain):
    name = "ecommerce"
    description = "Ecommerce analytics with ranked KPI fallback"
    required_columns = ["sessions"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "sessions" in df.columns or "orders" in df.columns

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

        if "conversion_rate" in kpis and kpis["conversion_rate"] < 0.02:
            insights.append({
                "level": "WARNING",
                "title": "Low Conversion Rate",
                "value": f"{kpis['conversion_rate']:.2%}",
                "so_what": "Low conversion indicates friction in the purchase journey."
            })

        if "cart_abandonment_rate" in kpis and kpis["cart_abandonment_rate"] > 0.6:
            insights.append({
                "level": "RISK",
                "title": "High Cart Abandonment",
                "value": f"{kpis['cart_abandonment_rate']:.1%}",
                "so_what": "High abandonment leads to lost revenue opportunities."
            })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "conversion_rate" in kpis and kpis["conversion_rate"] < 0.02:
            recs.append({
                "action": "Optimize checkout flow and page load speed",
                "expected_impact": "5–10% conversion uplift",
                "timeline": "2–4 weeks"
            })

        if "cart_abandonment_rate" in kpis and kpis["cart_abandonment_rate"] > 0.6:
            recs.append({
                "action": "Introduce cart recovery campaigns",
                "expected_impact": "10–20% cart recovery",
                "timeline": "1–2 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class EcommerceDomainDetector(BaseDomainDetector):
    domain_name = "ecommerce"

    ECOMMERCE_COLUMNS: Set[str] = {
        "sessions", "orders", "order_value", "revenue",
        "conversions", "carts", "checkouts",
        "returns", "payment_method", "traffic_source",
        "device", "lifetime_value", "ad_spend", "customers"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("ecommerce", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.ECOMMERCE_COLUMNS)

        score = min((len(matches) / len(self.ECOMMERCE_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="ecommerce",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="ecommerce",
        domain_cls=EcommerceDomain,
        detector_cls=EcommerceDomainDetector,
    )
