import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS (PURE, SAFE)
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> str | None:
    """
    Detect a usable time column without guessing.
    """
    for key in ["date", "period", "month", "year"]:
        col = resolve_column(df, key)
        if col:
            return col
    return None


def _prepare_time_series(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Safely sort dataframe by time column.
    """
    df_out = df.copy()
    try:
        df_out[time_col] = pd.to_datetime(df_out[time_col], errors="coerce")
        df_out = df_out.dropna(subset=[time_col])
        df_out = df_out.sort_values(time_col)
    except Exception:
        pass
    return df_out


# =====================================================
# FINANCE DOMAIN
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Defensible financial analytics (P&L, trends, ratios)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return resolve_column(df, "revenue") is not None or \
               resolve_column(df, "income") is not None

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        self.has_time_series = False

        if self.time_col:
            df = _prepare_time_series(df, self.time_col)
            self.has_time_series = df[self.time_col].nunique() >= 2

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        revenue = resolve_column(df, "revenue") or resolve_column(df, "income")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")
        profit = resolve_column(df, "profit")
        debt = resolve_column(df, "debt")
        equity = resolve_column(df, "equity")

        # --- Core KPIs ---
        if revenue and pd.api.types.is_numeric_dtype(df[revenue]):
            kpis["total_revenue"] = df[revenue].sum()

        if expense and pd.api.types.is_numeric_dtype(df[expense]):
            kpis["total_expense"] = df[expense].sum()

        if profit and pd.api.types.is_numeric_dtype(df[profit]):
            kpis["total_profit"] = df[profit].sum()

        # --- Derived KPIs ---
        if revenue and expense:
            r = df[revenue].sum()
            e = df[expense].sum()
            margin = _safe_div(r - e, r)
            if margin is not None:
                kpis["profit_margin"] = margin

        if debt and equity:
            d = df[debt].sum()
            e = df[equity].sum()
            ratio = _safe_div(d, e)
            if ratio is not None:
                kpis["debt_to_equity"] = ratio

        # --- Growth (only if time-series) ---
        if self.has_time_series and revenue:
            first = df[revenue].iloc[0]
            last = df[revenue].iloc[-1]
            growth = _safe_div(last - first, first)
            if growth is not None:
                kpis["revenue_growth"] = growth

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        def human_fmt(x, _):
            if x >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if x >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))

        revenue = resolve_column(df, "revenue") or resolve_column(df, "income")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")

        # --- Revenue Trend ---
        if self.has_time_series and revenue:
            p = output_dir / "revenue_trend.png"
            plt.figure(figsize=(6, 4))
            df[revenue].plot()
            plt.title("Revenue Trend")
            plt.ylabel("Revenue")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({
                "path": p,
                "caption": "Revenue trend over time"
            })

        # --- Expense vs Revenue ---
        if revenue and expense:
            p = output_dir / "revenue_vs_expense.png"
            plt.figure(figsize=(6, 4))
            df[[revenue, expense]].sum().plot(kind="bar")
            plt.title("Revenue vs Expense")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({
                "path": p,
                "caption": "Total revenue compared to expenses"
            })

        return visuals

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        margin = kpis.get("profit_margin")
        growth = kpis.get("revenue_growth")

        # --- Margin health ---
        if margin is not None and margin < 0.10:
            insights.append({
                "level": "WARNING",
                "title": "Low Profit Margin",
                "so_what": f"Profit margin is {margin:.1%}, indicating limited profitability."
            })

        # --- Growth vs Margin ---
        if growth is not None and margin is not None:
            if growth > 0 and margin < 0.15:
                insights.append({
                    "level": "RISK",
                    "title": "Growth Without Profitability",
                    "so_what": (
                        f"Revenue grew by {growth:.1%}, "
                        f"but profit margin remains low ({margin:.1%})."
                    )
                })

        # --- Fallback ---
        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Financial Performance Stable",
                "so_what": "No material financial risks detected based on available data."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []

        for i in self.generate_insights(df, kpis):
            if i["level"] == "RISK":
                recs.append({
                    "action": f"Immediate review required: {i['title']}",
                    "priority": "HIGH",
                    "timeline": "1–2 months",
                })
            elif i["level"] == "WARNING":
                recs.append({
                    "action": f"Monitor and optimize: {i['title']}",
                    "priority": "MEDIUM",
                    "timeline": "2–3 months",
                })

        if not recs:
            recs.append({
                "action": "Continue standard financial monitoring",
                "priority": "LOW",
                "timeline": "Ongoing",
            })

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"

    FINANCE_TOKENS: Set[str] = {
        "revenue", "income", "expense", "cost",
        "profit", "loss", "debt", "equity", "financial"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.FINANCE_TOKENS)]
        return DomainDetectionResult(
            domain="finance",
            confidence=min(len(hits) / 3, 1.0),
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="finance",
        domain_cls=FinanceDomain,
        detector_cls=FinanceDomainDetector,
    )
