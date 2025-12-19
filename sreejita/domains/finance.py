import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS (PURE + DEFENSIVE)
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Finance-safe time detector:
    - Explicit finance names first
    - Then validated 'date' containment
    """

    explicit = [
        "fiscal date", "posting date", "transaction date",
        "invoice date", "date", "period", "month", "year"
    ]

    cols_lower = {c.lower(): c for c in df.columns}

    # Pass 1 — explicit
    for key in explicit:
        for low, real in cols_lower.items():
            if key == low and not df[real].isna().all():
                try:
                    pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                    return real
                except Exception:
                    continue

    # Pass 2 — fuzzy
    for low, real in cols_lower.items():
        if "date" in low and not df[real].isna().all():
            try:
                pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                return real
            except Exception:
                continue

    return None


def _prepare_time_series(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df_out = df.copy()
    try:
        df_out[time_col] = pd.to_datetime(df_out[time_col], errors="coerce")
        df_out = df_out.dropna(subset=[time_col])
        df_out = df_out.sort_values(time_col)
    except Exception:
        pass
    return df_out


# =====================================================
# FINANCE DOMAIN (v2.x)
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Defensible financial analytics (Revenue, Expense, Budget, Ratios)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return any(
            resolve_column(df, k) is not None
            for k in ["revenue", "income", "sales", "expense", "cost"]
        )

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

        revenue = (
            resolve_column(df, "revenue")
            or resolve_column(df, "income")
            or resolve_column(df, "sales")
        )
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")
        profit = resolve_column(df, "profit") or resolve_column(df, "net income")
        budget = resolve_column(df, "budget") or resolve_column(df, "target")

        if revenue and pd.api.types.is_numeric_dtype(df[revenue]):
            kpis["total_revenue"] = df[revenue].sum()

        if expense and pd.api.types.is_numeric_dtype(df[expense]):
            kpis["total_expense"] = df[expense].sum()

        if profit and pd.api.types.is_numeric_dtype(df[profit]):
            kpis["total_profit"] = df[profit].sum()
        elif "total_revenue" in kpis and "total_expense" in kpis:
            kpis["total_profit"] = kpis["total_revenue"] - kpis["total_expense"]

        if budget and revenue and pd.api.types.is_numeric_dtype(df[budget]):
            planned = df[budget].sum()
            if planned != 0:
                actual = df[revenue].sum()
                var = actual - planned
                kpis["budget_variance_abs"] = var
                kpis["budget_variance_pct"] = _safe_div(var, planned)

        if "total_profit" in kpis and "total_revenue" in kpis:
            kpis["profit_margin"] = _safe_div(
                kpis["total_profit"], kpis["total_revenue"]
            )

        if self.has_time_series and revenue and df[revenue].notna().sum() >= 2:
            first = df[revenue].iloc[0]
            last = df[revenue].iloc[-1]
            if first != 0:
                kpis["revenue_growth"] = _safe_div(last - first, first)

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        def human_fmt(x, _):
            if abs(x) >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))

        revenue = (
            resolve_column(df, "revenue")
            or resolve_column(df, "income")
            or resolve_column(df, "sales")
        )
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")

        # 1️⃣ Revenue Trend
        if self.has_time_series and revenue:
            p = output_dir / "revenue_trend.png"
            plt.figure(figsize=(7, 4))

            plot_df = df.copy()
            if len(df) > 100 and pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("M")
                    .sum()
                    .reset_index()
                )

            plt.plot(plot_df[self.time_col], plot_df[revenue], linewidth=2)
            plt.title("Revenue Trend")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Revenue performance over time"
            })

        # 2️⃣ P&L Summary
        if revenue and expense:
            p = output_dir / "pnl_summary.png"
            rev = df[revenue].sum()
            exp = df[expense].sum()
            prof = rev - exp

            steps = ["Revenue", "Expenses", "Net Result"]
            values = [rev, -exp, prof]
            bottoms = [0, rev, 0]

            plt.figure(figsize=(7, 4))
            bars = plt.bar(steps, values, bottom=bottoms)
            for bar, val in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    human_fmt(val, None),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

            plt.title("P&L Summary")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Revenue to net result summary"
            })

        # 3️⃣ Expense Breakdown (Optional but Safe)
        if expense:
            category = (
                resolve_column(df, "department")
                or resolve_column(df, "category")
                or resolve_column(df, "account")
            )

            p = output_dir / "expense_breakdown.png"
            plt.figure(figsize=(7, 4))

            if category:
                grp = (
                    df.groupby(category)[expense]
                    .sum()
                    .sort_values(ascending=False)
                    .head(7)
                )
                grp.plot(kind="bar")
                plt.title("Top Expense Contributors")
                plt.xticks(rotation=45, ha="right")
            else:
                plt.bar(["Total Expenses"], [df[expense].sum()])
                plt.title("Total Expenses")

            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Expense contribution overview"
            })

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        margin = kpis.get("profit_margin")
        var_pct = kpis.get("budget_variance_pct")

        if margin is not None:
            if margin < 0:
                insights.append({
                    "level": "RISK",
                    "title": "High Burn Rate Detected",
                    "so_what": f"Expenses exceed revenue. Margin is {margin:.1%}."
                })
            elif margin < 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Low Profit Margin",
                    "so_what": f"Margin is {margin:.1%}."
                })

        if var_pct is not None:
            if var_pct < -0.05:
                insights.append({
                    "level": "RISK",
                    "title": "Missed Budget Target",
                    "so_what": f"Revenue is {abs(var_pct):.1%} below plan."
                })
            elif var_pct > 0.10:
                insights.append({
                    "level": "INFO",
                    "title": "Exceeding Budget",
                    "so_what": f"Revenue exceeds plan by {var_pct:.1%}."
                })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Financial Performance Stable",
                "so_what": "Key financial indicators are within expected ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        for i in self.generate_insights(df, kpis):
            if i["level"] == "RISK":
                recs.append({
                    "action": f"URGENT: {i['title']}",
                    "priority": "HIGH",
                    "timeline": "Immediate",
                })
            elif i["level"] == "WARNING":
                recs.append({
                    "action": f"Investigate: {i['title']}",
                    "priority": "MEDIUM",
                    "timeline": "This Quarter",
                })

        if not recs:
            recs.append({
                "action": "Maintain current financial controls",
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
        "revenue", "income", "sales",
        "expense", "cost", "spend",
        "budget", "forecast",
        "profit", "loss",
        "ledger", "fiscal", "amount"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.FINANCE_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        return DomainDetectionResult(
            domain="finance",
            confidence=confidence,
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
