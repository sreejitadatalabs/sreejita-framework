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
    Detect a real time column safely (Datetime, Numeric, or String-Date).
    """
    # 1. Prioritize explicit names to narrow search space
    for key in ["date", "period", "month", "year"]:
        col = resolve_column(df, key)
        if not col:
            continue
        
        # 2. Skip if column is entirely empty (prevents crashes later)
        if df[col].isna().all():
            continue

        # A. Accept standard datetime columns directly
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

        # B. Accept numeric Year/Month (with range guards)
        if pd.api.types.is_numeric_dtype(df[col]):
            # usage of unique() speeds up large column checks
            values = df[col].dropna().unique()
            
            if len(values) == 0: continue

            v_min, v_max = values.min(), values.max()

            # Year-like (e.g. 1950–2050)
            if 1950 <= v_min and v_max <= 2050:
                return col

            # Month-like (1–12)
            # FIX 2: Added nunique check to avoid confusing category codes with months
            if 1 <= v_min and v_max <= 12 and len(values) <= 12:
                return col

        # C. Accept String Dates (The Missing Piece)
        # Often "date" columns in CSVs are just strings: "2023-01-01"
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            try:
                # Test the first 10 non-null values to see if they parse
                sample = df[col].dropna().iloc[:10]
                pd.to_datetime(sample, errors="raise") # Will raise error if not date-like
                return col
            except (ValueError, TypeError):
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
# FINANCE DOMAIN
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Defensible financial analytics (P&L, Budget, Ratios)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return (
            resolve_column(df, "revenue") is not None
            or resolve_column(df, "income") is not None
        )

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        self.has_time_series = False

        if self.time_col:
            df = _prepare_time_series(df, self.time_col)
            # Need at least 2 points to draw a line
            self.has_time_series = df[self.time_col].nunique() >= 2

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        revenue = resolve_column(df, "revenue") or resolve_column(df, "income")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")
        profit = resolve_column(df, "profit")
        budget = resolve_column(df, "budget") or resolve_column(df, "target")
        debt = resolve_column(df, "debt")
        equity = resolve_column(df, "equity")

        # ---- Totals ----
        if revenue and pd.api.types.is_numeric_dtype(df[revenue]):
            kpis["total_revenue"] = df[revenue].sum()

        if expense and pd.api.types.is_numeric_dtype(df[expense]):
            kpis["total_expense"] = df[expense].sum()

        if profit and pd.api.types.is_numeric_dtype(df[profit]):
            kpis["total_profit"] = df[profit].sum()
        elif "total_revenue" in kpis and "total_expense" in kpis:
            kpis["total_profit"] = kpis["total_revenue"] - kpis["total_expense"]

        # ---- Budget Variance ----
        if budget and revenue:
            actual = df[revenue].sum()
            planned = df[budget].sum()
            var = actual - planned
            kpis["budget_variance_abs"] = var
            kpis["budget_variance_pct"] = _safe_div(var, planned)

        # ---- Ratios ----
        if "total_profit" in kpis and "total_revenue" in kpis:
            kpis["profit_margin"] = _safe_div(
                kpis["total_profit"], kpis["total_revenue"]
            )

        if debt and equity:
            kpis["debt_to_equity"] = _safe_div(
                df[debt].sum(), df[equity].sum()
            )

        # ---- Growth (STRICT) ----
        # Only calculate if we have time series AND enough data points
        if (
            self.has_time_series
            and revenue
            and df[revenue].notna().sum() >= 2
        ):
            first = df[revenue].iloc[0]
            last = df[revenue].iloc[-1]
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

        revenue = resolve_column(df, "revenue") or resolve_column(df, "income")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")
        budget = resolve_column(df, "budget")

        # ---- Revenue Trend (ONLY if time-series) ----
        if self.has_time_series and revenue:
            p = output_dir / "revenue_trend.png"
            plt.figure(figsize=(7, 4))
            plt.plot(df[self.time_col], df[revenue], label="Revenue", linewidth=2)
            
            # Show budget line only if data density is reasonable (<50 points)
            if budget and len(df) < 50:
                plt.plot(
                    df[self.time_col],
                    df[budget],
                    linestyle="--",
                    label="Budget",
                    alpha=0.7,
                )
            plt.title("Revenue Trend")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.legend()
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({
                "path": p,
                "caption": "Revenue performance over time"
            })

        # ---- P&L Waterfall (GATED & SAFE) ----
        # FIX 3: Added checks ensuring columns are numeric to prevent crashes
        if (
            revenue and expense 
            and len(df) <= 12
            and pd.api.types.is_numeric_dtype(df[revenue])
            and pd.api.types.is_numeric_dtype(df[expense])
        ):
            p = output_dir / "pnl_waterfall.png"

            rev = df[revenue].sum()
            exp = df[expense].sum()
            profit = rev - exp

            steps = ["Revenue", "Expenses", "Net Profit"]
            values = [rev, -exp, profit]
            bottoms = [0, rev, 0]
            colors = ["#2ca02c", "#d62728", "#1f77b4"]

            plt.figure(figsize=(7, 4))
            bars = plt.bar(steps, values, bottom=bottoms, color=colors)
            
            # Add text labels on bars
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
            
            plt.title("P&L Waterfall")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({
                "path": p,
                "caption": "Bridge from revenue to net profit"
            })

        return visuals

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        margin = kpis.get("profit_margin")
        growth = kpis.get("revenue_growth")
        var_pct = kpis.get("budget_variance_pct")

        # ---- Margin ----
        if margin is not None and margin < 0.10:
            insights.append({
                "level": "WARNING",
                "title": "Low Profit Margin",
                "so_what": f"Profit margin is {margin:.1%}, indicating cost pressure."
            })

        # ---- Budget Variance ----
        if var_pct is not None:
            # We treat missed budget as RISK only if we are confident in the time series
            if var_pct < -0.05:
                level = "RISK" if self.has_time_series else "INFO"
                insights.append({
                    "level": level,
                    "title": "Missed Budget Target",
                    "so_what": f"Revenue is {abs(var_pct):.1%} below budget."
                })
            elif var_pct > 0.10:
                insights.append({
                    "level": "INFO",
                    "title": "Exceeding Budget",
                    "so_what": f"Revenue exceeds budget by {var_pct:.1%}."
                })

        # ---- Hollow Growth ----
        if growth is not None and margin is not None:
            if growth > 0.20 and margin < 0.05:
                insights.append({
                    "level": "RISK",
                    "title": "Unprofitable Growth",
                    "so_what": (
                        f"High growth ({growth:.1%}) with thin margins "
                        f"({margin:.1%})."
                    )
                })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Financial Performance Stable",
                "so_what": "Key financial indicators are within expected ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        recs: List[Dict[str, Any]] = []

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
        "revenue", "income", "expense", "cost", "cogs",
        "profit", "loss", "debt", "equity", "budget", "forecast"
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
