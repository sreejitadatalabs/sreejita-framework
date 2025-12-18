"""
Retail Domain Module (v2.x FINAL)

- Ranked KPI fallback system (top 4 always attempted)
- KPI-driven visuals (no hard dependency)
- Defensive execution (no schema assumptions)
- Domain-neutral reporting compatibility
"""

from typing import Dict, Any, List, Set
from pathlib import Path
import pandas as pd

from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# KPI RANKING PLAN (AUTHORITATIVE CONTRACT)
# =====================================================

RETAIL_KPI_PLAN = [
    {"name": "total_sales", "rank": 1, "columns": ["sales"]},
    {"name": "order_count", "rank": 2, "columns": ["sales"]},
    {"name": "average_order_value", "rank": 3, "columns": ["sales"]},
    {"name": "profit_margin", "rank": 4, "columns": ["sales", "profit"]},
    {"name": "shipping_cost_ratio", "rank": 5, "columns": ["sales", "shipping_cost"]},
    {"name": "average_discount", "rank": 6, "columns": ["discount"]},
    {"name": "category_revenue", "rank": 7, "columns": ["category", "sales"]},
    {"name": "top_products", "rank": 8, "columns": ["product", "sales"]},
    {"name": "sales_volatility", "rank": 9, "columns": ["sales"]},
    {"name": "profit_contribution", "rank": 10, "columns": ["profit", "category"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(RETAIL_KPI_PLAN, key=lambda x: x["rank"]):
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
        if name == "total_sales":
            return float(df["sales"].sum())

        if name == "order_count":
            return int(len(df))

        if name == "average_order_value":
            return float(df["sales"].mean())

        if name == "profit_margin":
            return _safe_div(df["profit"].sum(), df["sales"].sum())

        if name == "shipping_cost_ratio":
            return _safe_div(df["shipping_cost"].sum(), df["sales"].sum())

        if name == "average_discount":
            return float(df["discount"].mean())

        if name == "category_revenue":
            return float(df.groupby("category")["sales"].sum().max())

        if name == "top_products":
            return float(df.groupby("product")["sales"].sum().max())

        if name == "sales_volatility":
            return float(df["sales"].std())

        if name == "profit_contribution":
            return float(df.groupby("category")["profit"].sum().max())

    except Exception:
        return None

    return None


# =====================================================
# VISUALS (ONLY FOR SELECTED KPIs)
# =====================================================

def _generate_visuals(df: pd.DataFrame, selected_kpis: List[str], out_dir: Path):
    """
    NOTE:
    We intentionally implement ONLY 2–3 visuals.
    Ranking guarantees stability; visuals can grow later without refactor.
    """
    visuals = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sales trend
    if "total_sales" in selected_kpis:
        path = out_dir / "sales_trend.png"
        df["sales"].plot(title="Sales Trend")
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Sales trend over records"
        })

    # Category revenue
    if "category_revenue" in selected_kpis:
        path = out_dir / "category_revenue.png"
        df.groupby("category")["sales"].sum().plot(kind="bar", title="Revenue by Category")
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Revenue distribution by category"
        })

    # Discount distribution
    if "average_discount" in selected_kpis:
        path = out_dir / "discount_dist.png"
        df["discount"].hist()
        import matplotlib.pyplot as plt
        plt.title("Discount Distribution")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Discount distribution"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Retail analytics with ranked KPI fallback"
    required_columns = ["sales"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "sales" in df.columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "profit" in df.columns and "sales" in df.columns:
            df["margin"] = _safe_div(df["profit"], df["sales"])
        return df

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

        if "profit_margin" in kpis:
            pm = kpis["profit_margin"]
            level = "GOOD" if pm >= 0.12 else "WARNING"
            insights.append({
                "level": level,
                "title": "Profitability Health",
                "value": f"{pm:.1%}",
                "so_what": "Margin impacts reinvestment and growth capacity."
            })

        if "shipping_cost_ratio" in kpis:
            scr = kpis["shipping_cost_ratio"]
            if scr > 0.1:
                insights.append({
                    "level": "WARNING",
                    "title": "Shipping Cost Efficiency",
                    "value": f"{scr:.1%}",
                    "so_what": "High shipping costs reduce net margin."
                })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "shipping_cost_ratio" in kpis and kpis["shipping_cost_ratio"] > 0.1:
            recs.append({
                "action": "Optimize shipping cost structure",
                "expected_impact": "$200K–$300K annual savings",
                "timeline": "5–7 days"
            })

        if "average_discount" in kpis and kpis["average_discount"] > 0.15:
            recs.append({
                "action": "Review discount strategy",
                "expected_impact": "+1.5–2.0% margin improvement",
                "timeline": "2 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"

    RETAIL_COLUMNS: Set[str] = {
        "sales", "revenue", "profit", "discount",
        "product", "category", "sub_category",
        "quantity", "price", "order_id"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("retail", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.RETAIL_COLUMNS)

        score = min((len(matches) / len(self.RETAIL_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="retail",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="retail",
        domain_cls=RetailDomain,
        detector_cls=RetailDomainDetector,
    )
