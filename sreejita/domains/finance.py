"""
Finance Domain Module (v2.x FINAL)

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

FINANCE_KPI_PLAN = [
    {"name": "total_revenue", "rank": 1, "columns": ["revenue"]},
    {"name": "net_income", "rank": 2, "columns": ["net_income"]},
    {"name": "profit_margin", "rank": 3, "columns": ["net_income", "revenue"]},
    {"name": "cash_flow", "rank": 4, "columns": ["cash_flow"]},
    {"name": "ebitda_margin", "rank": 5, "columns": ["ebitda", "revenue"]},
    {"name": "expense_ratio", "rank": 6, "columns": ["expenses", "revenue"]},
    {"name": "return_on_assets", "rank": 7, "columns": ["net_income", "assets"]},
    {"name": "debt_to_equity", "rank": 8, "columns": ["liabilities", "equity"]},
    {"name": "revenue_growth", "rank": 9, "columns": ["revenue"]},
    {"name": "liquidity_ratio", "rank": 10, "columns": ["current_assets", "current_liabilities"]},
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _select_ranked_kpis(df: pd.DataFrame, max_kpis: int = 4) -> List[str]:
    selected = []
    for item in sorted(FINANCE_KPI_PLAN, key=lambda x: x["rank"]):
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
        if name == "total_revenue":
            return float(df["revenue"].sum())

        if name == "net_income":
            return float(df["net_income"].sum())

        if name == "profit_margin":
            return _safe_div(df["net_income"].sum(), df["revenue"].sum())

        if name == "cash_flow":
            return float(df["cash_flow"].sum())

        if name == "ebitda_margin":
            return _safe_div(df["ebitda"].sum(), df["revenue"].sum())

        if name == "expense_ratio":
            return _safe_div(df["expenses"].sum(), df["revenue"].sum())

        if name == "return_on_assets":
            return _safe_div(df["net_income"].sum(), df["assets"].mean())

        if name == "debt_to_equity":
            return _safe_div(df["liabilities"].mean(), df["equity"].mean())

        if name == "revenue_growth":
            if len(df) < 2:
                return None
            first = df["revenue"].iloc[0]
            last = df["revenue"].iloc[-1]
            return _safe_div(last - first, first)

        if name == "liquidity_ratio":
            return _safe_div(
                df["current_assets"].mean(),
                df["current_liabilities"].mean()
            )

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

    if "total_revenue" in selected_kpis:
        path = out_dir / "revenue_trend.png"
        df["revenue"].plot()
        plt.title("Revenue Trend")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Revenue trend over time"
        })

    if "profit_margin" in selected_kpis:
        path = out_dir / "profit_margin.png"
        (_safe_div(df["net_income"], df["revenue"])).plot()
        plt.title("Profit Margin Trend")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Profit margin trend"
        })

    if "expense_ratio" in selected_kpis:
        path = out_dir / "expense_ratio.png"
        (_safe_div(df["expenses"], df["revenue"])).plot()
        plt.title("Expense Ratio Trend")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        visuals.append({
            "path": path,
            "caption": "Expense ratio over time"
        })

    return visuals[:4]


# =====================================================
# DOMAIN CLASS
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Finance analytics with ranked KPI fallback"
    required_columns = ["revenue"]

    def validate_data(self, df: pd.DataFrame) -> bool:
        return "revenue" in df.columns

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

        if "profit_margin" in kpis and kpis["profit_margin"] < 0.1:
            insights.append({
                "level": "WARNING",
                "title": "Low Profit Margin",
                "value": f"{kpis['profit_margin']:.1%}",
                "so_what": "Low margins reduce financial resilience."
            })

        if "debt_to_equity" in kpis and kpis["debt_to_equity"] > 2.0:
            insights.append({
                "level": "RISK",
                "title": "High Leverage",
                "value": f"{kpis['debt_to_equity']:.2f}",
                "so_what": "High leverage increases financial risk."
            })

        return insights

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if "expense_ratio" in kpis and kpis["expense_ratio"] > 0.7:
            recs.append({
                "action": "Reduce operational expenses",
                "expected_impact": "5–10% improvement in net margin",
                "timeline": "4–8 weeks"
            })

        if "debt_to_equity" in kpis and kpis["debt_to_equity"] > 2.0:
            recs.append({
                "action": "Review capital structure and debt levels",
                "expected_impact": "Improved financial stability",
                "timeline": "6–12 weeks"
            })

        return recs

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        selected = _select_ranked_kpis(df)
        return _generate_visuals(df, selected, output_dir)


# =====================================================
# DETECTOR (UNCHANGED, CORRECT)
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"

    FINANCE_COLUMNS: Set[str] = {
        "revenue", "net_income", "cash_flow", "ebitda",
        "expenses", "assets", "liabilities", "equity",
        "current_assets", "current_liabilities"
    }

    def detect(self, df) -> DomainDetectionResult:
        if df is None or not hasattr(df, "columns"):
            return DomainDetectionResult("finance", 0.0, {"reason": "invalid_df"})

        cols = {str(c).lower() for c in df.columns}
        matches = cols.intersection(self.FINANCE_COLUMNS)

        score = min((len(matches) / len(self.FINANCE_COLUMNS)) * 1.5, 1.0)

        return DomainDetectionResult(
            domain="finance",
            confidence=score,
            signals={"matched_columns": list(matches)}
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register(
        name="finance",
        domain_cls=FinanceDomain,
        detector_cls=FinanceDomainDetector,
    )
