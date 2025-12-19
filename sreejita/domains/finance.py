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
# FINANCE DOMAIN (v2.x GOLD STANDARD)
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

        # 1. Base Totals
        if revenue and pd.api.types.is_numeric_dtype(df[revenue]):
            kpis["total_revenue"] = df[revenue].sum()

        if expense and pd.api.types.is_numeric_dtype(df[expense]):
            kpis["total_expense"] = df[expense].sum()

        if profit and pd.api.types.is_numeric_dtype(df[profit]):
            kpis["total_profit"] = df[profit].sum()
        elif "total_revenue" in kpis and "total_expense" in kpis:
            kpis["total_profit"] = kpis["total_revenue"] - kpis["total_expense"]

        # 2. Budget Logic
        if budget and revenue and pd.api.types.is_numeric_dtype(df[budget]):
            actual = df[revenue].sum()
            planned = df[budget].sum()
            if planned != 0:
                var = actual - planned
                kpis["budget_variance_abs"] = var
                kpis["budget_variance_pct"] = _safe_div(var, planned)

        # 3. Ratios & Thresholds
        if "total_profit" in kpis and "total_revenue" in kpis:
            kpis["profit_margin"] = _safe_div(
                kpis["total_profit"], kpis["total_revenue"]
            )
            # Healthy standard margin is often > 15%
            kpis["target_profit_margin"] = 0.15 

        if "total_expense" in kpis and "total_revenue" in kpis:
            kpis["expense_ratio"] = _safe_div(kpis["total_expense"], kpis["total_revenue"])
            kpis["target_expense_ratio"] = 0.70 # Target < 70%

        # 4. Growth Logic
        if self.has_time_series and revenue and df[revenue].notna().sum() >= 2:
            first = df[revenue].iloc[0]
            last = df[revenue].iloc[-1]
            if first != 0:
                kpis["revenue_growth"] = _safe_div(last - first, first)

        return kpis

    # ---------------- VISUALS (MAX 4, DATA-DRIVEN) ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- FIX 2: Calculate KPIs once to prevent re-computation ---
        kpis = self.calculate_kpis(df)

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

        # -------- VISUAL 1: Revenue Trend --------
        if self.has_time_series and revenue:
            p = output_dir / "revenue_trend.png"
            plt.figure(figsize=(7, 4))

            plot_df = df.copy()
            # Smart Aggregation: Only if huge (>100) and confirmed datetime
            if len(df) > 100 and pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME") 
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

        # -------- VISUAL 2: P&L Summary --------
        if revenue and expense:
            p = output_dir / "pnl_summary.png"
            rev = df[revenue].sum()
            exp = df[expense].sum()
            prof = rev - exp

            steps = ["Revenue", "Expenses", "Net Result"]
            values = [rev, -exp, prof]
            bottoms = [0, rev, 0]
            colors = ["#2ca02c", "#d62728", "#1f77b4"]

            plt.figure(figsize=(7, 4))
            bars = plt.bar(steps, values, bottom=bottoms, color=colors)
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

        # -------- VISUAL 3: Expense Breakdown --------
        # --- FIX 3: Defensive check for numeric expense ---
        if expense and pd.api.types.is_numeric_dtype(df[expense]):
            p = output_dir / "expense_breakdown.png"
            plt.figure(figsize=(7, 4))

            category = (
                resolve_column(df, "department")
                or resolve_column(df, "category")
                or resolve_column(df, "account")
            )

            if category:
                grp = (
                    df.groupby(category)[expense]
                    .sum()
                    .sort_values(ascending=False)
                    .head(7)
                )
                grp.plot(kind="bar", color="#d62728")
                plt.title("Top Expense Contributors")
                plt.xticks(rotation=45, ha='right')
            else:
                plt.bar(["Total Expenses"], [df[expense].sum()], color="#d62728")
                plt.title("Total Expenses")

            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Expense contribution overview"
            })

        # -------- VISUAL 4: Financial Ratios --------
        # --- FIX 2: Use pre-calculated KPIs ---
        ratios = {
            "Profit Margin": kpis.get("profit_margin"),
            "Budget Variance %": kpis.get("budget_variance_pct"),
            "Revenue Growth": kpis.get("revenue_growth"),
        }

        ratios = {k: v for k, v in ratios.items() if v is not None}

        if ratios:
            p = output_dir / "financial_ratios.png"
            plt.figure(figsize=(7, 4))
            plt.bar(ratios.keys(), ratios.values(), color="#1f77b4")
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title("Key Financial Ratios")
            plt.gca().yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:.0%}")
            )
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Snapshot of key financial ratios"
            })

        return visuals[:4]

    # ---------------- INSIGHTS (ENHANCED) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        margin = kpis.get("profit_margin")
        var_pct = kpis.get("budget_variance_pct")
        growth = kpis.get("revenue_growth")
        expense_ratio = kpis.get("expense_ratio")

        # 1. Profitability (Burn Rate vs Healthy)
        if margin is not None:
            if margin < 0:
                insights.append({
                    "level": "RISK",
                    "title": "High Burn Rate Detected",
                    "so_what": f"Expenses exceed revenue. Net margin is {margin:.1%}."
                })
            elif margin < 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Low Profit Margin",
                    "so_what": f"Margin is {margin:.1%}, below the healthy target of 15%."
                })
        
        # 2. Efficiency Check (Corrected Thresholds)
        # > 85% is High Risk, > 70% is Warning
        if expense_ratio is not None:
            if expense_ratio > 0.85:
                insights.append({
                    "level": "RISK",
                    "title": "Critical Operational Costs",
                    "so_what": f"Expenses consume {expense_ratio:.1%} of revenue, threatening solvency."
                })
            elif expense_ratio > 0.70:
                insights.append({
                    "level": "WARNING",
                    "title": "High Operational Costs",
                    "so_what": f"Expenses are consuming {expense_ratio:.1%} of total revenue."
                })

        # 3. Budget Check
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

        # 4. Growth Check
        if growth is not None:
            if growth < 0:
                insights.append({
                    "level": "WARNING",
                    "title": "Revenue Contraction",
                    "so_what": f"Revenue has declined by {abs(growth):.1%} over the period."
                })
            elif growth > 0.20:
                insights.append({
                    "level": "INFO",
                    "title": "High Growth Trajectory",
                    "so_what": f"Revenue grew by {growth:.1%}, indicating strong momentum."
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
