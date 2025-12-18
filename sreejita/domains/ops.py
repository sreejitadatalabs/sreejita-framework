"""
Operations / Logistics Domain Module (v2.x FINAL)

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

OPS_KPI_PLAN = [
    {"name": "on_time_delivery_rate", "rank": 1, "columns": ["on_time_delivery"]},
    {"name": "avg_delivery_time", "rank": 2, "columns": ["delivery_time"]},
    {"name": "order_volume", "rank": 3, "columns": ["order_id"]},
    {"name": "inventory_turnover", "rank": 4, "columns": ["inventory_level", "orders"]},
    {"name": "return_rate", "rank": 5, "columns": ["returns", "orders"]},
    {"name": "avg_processing_time", "rank": 6, "columns": ["processing_time"]},
    {"name": "supplier_count", "rank": 7, "columns": ["supplier"]},
    {"name": "warehouse_utilization", "rank": 8, "columns": ["warehouse", "capacity_used"]},
    {"name": "shipping_mode_mix", "rank": 9, "columns": ["shipping_mode"]},
    {"name": "location_mix", "rank": 10, "columns": ["location"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(OPS_KPI_PLAN, key=lambda x: x["rank"]):
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
        if name == "on_time_delivery_rate":
            return float(df["on_time_delivery"].mean())

        if name == "avg_delivery_time":
            return float(df["delivery_time"].mean())

        if name == "order_volume":
            return int(df["order_id"].nunique())

        if name == "inventory_turnover":
            return _safe_div(df["orders"].sum(), df["inventory_level"].mean())

        if name == "return_rate":
            return _safe_div(df["returns"].sum(), df["orders"].sum())

        if name == "avg_processing_time":
            return float(df["processing_time"].mean())

        if name == "supplier_count":
            return int(df["supplier"].nunique())

        if name == "warehouse_utilization":
            return float(df["capacity_used"].mean())

        if name == "shipping_mode_mix":
            return df["shipping_mode"].value_counts(normalize=True).iloc[0]

        if name == "location_mix":
            return df["location"].value_counts(normalize=True).iloc[0]

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

    if "on_time_delivery_rate" in selected_kpis:
        path = out_dir / "on_time_delivery.png"
        df["on_time_delivery"].value_counts().plot(kind="bar")
        plt.title("On-Time Delivery Performance")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "On-time vs delayed delivery distribution"
        })

    if "avg_delivery_time" in selected_kpis:
        path = out_dir / "delivery_time.png"
        df["delivery_time"].hist()
        plt.title("Delivery Time Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of delivery times"
        })

    if "shipping_mode_mix" in selected_kpis:
        path = out_dir / "shipping_mode_mix.png"
        df["shipping_mode"].value_counts().plot(kind="bar")
        plt.title("Shipping Mode Mix")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Distribution of shipping modes"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class OpsDomain(BaseDomain):
    name = "ops"
    description = "Operations & logistics analytics with ranked KPI fallback"
    required_columns = ["order_id"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "order_id" in df.columns

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

        if "on_time_delivery_rate" in kpis and kpis["on_time_delivery_rate"] < 0.9:
            insights.append({
                "level": "RISK",
                "title": "Low On-Time Delivery Rate",
                "value": f"{kpis['on_time_delivery_rate']:.1%}",
                "so_what": "Late deliveries impact customer satisfaction and SLA compliance."
            })

        if "avg_delivery_time" in kpis and kpis["avg_delivery_time"] > 7:
            insights.append({
                "level": "WARNING",
                "title": "Slow Delivery Times",
                "value": f"{kpis['avg_delivery_time']:.1f} days",
                "so_what": "Long delivery times reduce competitiveness."
            })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "on_time_delivery_rate" in kpis and kpis["on_time_delivery_rate"] < 0.9:
            recs.append({
                "action": "Review carrier performance and routing strategy",
                "expected_impact": "5–10% improvement in on-time delivery",
                "timeline": "3–5 weeks"
            })

        if "avg_delivery_time" in kpis and kpis["avg_delivery_time"] > 7:
            recs.append({
                "action": "Optimize warehouse dispatch and fulfillment process",
                "expected_impact": "1–2 day delivery time reduction",
                "timeline": "4–6 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class OpsDomainDetector(BaseDomainDetector):
    domain_name = "ops"

    OPS_COLUMNS: Set[str] = {
        "order_id", "delivery_time", "on_time_delivery",
        "inventory_level", "orders", "returns",
        "processing_time", "supplier",
        "warehouse", "capacity_used",
        "shipping_mode", "location"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("ops", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.OPS_COLUMNS)

        score = min((len(matches) / len(self.OPS_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="ops",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="ops",
        domain_cls=OpsDomain,
        detector_cls=OpsDomainDetector,
    )
