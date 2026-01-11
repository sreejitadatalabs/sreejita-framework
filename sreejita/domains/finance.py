# =====================================================
# Finance Domain — Block 1
# Imports · Helpers · Time Detection
# =====================================================

from __future__ import annotations

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import warnings

from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

warnings.filterwarnings("ignore")


# =====================================================
# GENERIC SAFE HELPERS (Framework Standard)
# =====================================================

def safe_divide(n: float | pd.Series, d: float | pd.Series) -> float | pd.Series:
    """
    Division helper with strict zero / null protection.
    Returns NaN (not 0, not inf) to preserve honesty.
    """
    return np.where(
        (d == 0) | pd.isna(d),
        np.nan,
        n / d
    )


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Coerce columns to numeric safely.
    Non-parsable values become NaN.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =====================================================
# TIME DETECTION (Finance-Aware, Boundary-Safe)
# =====================================================

@dataclass
class TimeContext:
    time_column: Optional[str]
    granularity: str           # monthly | quarterly | yearly | irregular | none
    is_ordered: bool
    coverage_periods: int


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detects a likely time column using finance-safe heuristics.
    No forced parsing, no assumptions.
    """
    candidates = [
        "date", "timestamp", "period",
        "month", "year", "quarter",
        "fiscal_date", "fiscal_period",
        "reporting_date"
    ]

    for col in df.columns:
        lcol = col.lower()
        if any(k in lcol for k in candidates):
            try:
                pd.to_datetime(df[col].dropna().iloc[0])
                return col
            except Exception:
                continue

    # Fallback: true datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    return None


def infer_time_granularity(series: pd.Series) -> str:
    """
    Infers time granularity without enforcing structure.
    """
    if series.isna().all():
        return "none"

    s = series.astype(str)

    if s.str.fullmatch(r"\d{4}").all():
        return "yearly"

    if s.str.contains("Q", case=False).any():
        return "quarterly"

    if series.nunique(dropna=True) >= 3:
        return "monthly"

    return "irregular"


def build_time_context(df: pd.DataFrame) -> TimeContext:
    """
    Builds a degradation-safe TimeContext object.
    """
    time_col = detect_time_column(df)

    if time_col is None:
        return TimeContext(
            time_column=None,
            granularity="none",
            is_ordered=False,
            coverage_periods=0
        )

    series = df[time_col]

    try:
        parsed = pd.to_datetime(series, errors="coerce")
        is_ordered = parsed.is_monotonic_increasing
    except Exception:
        is_ordered = False

    return TimeContext(
        time_column=time_col,
        granularity=infer_time_granularity(series),
        is_ordered=is_ordered,
        coverage_periods=int(series.nunique(dropna=True))
    )


# =====================================================
# FINANCE-SPECIFIC ANALYTIC HELPERS (OPTIONAL SIGNALS)
# =====================================================

def benford_deviation(series: pd.Series) -> float:
    """
    Calculates deviation from Benford's Law.
    Used ONLY as a weak risk / anomaly proxy.
    Returns 0.0 if data is insufficient.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return 0.0

    s = series.dropna().astype(str)
    first_digits = (
        s.str.lstrip("-")
         .str.replace(".", "", regex=False)
         .str[0]
    )

    first_digits = first_digits[first_digits.str.isnumeric()]

    # Governance: require sufficient volume
    if len(first_digits) < 100:
        return 0.0

    observed = first_digits.value_counts(normalize=True)

    benford = {
        str(d): np.log10(1 + 1 / d)
        for d in range(1, 10)
    }

    deviation = sum(
        abs(observed.get(str(d), 0) - benford[str(d)])
        for d in range(1, 10)
    )

    return float(deviation)


# =====================================================
# VISUAL FORMATTERS (NO PLOTTING YET)
# =====================================================

def human_currency_formatter(x, _):
    """
    Converts large numeric values into human-readable format.
    """
    if pd.isna(x):
        return ""
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{x:.0f}"

# =====================================================
# Finance Domain — Block 2
# Domain Class · Preprocess · Signal Validation
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Universal Finance Intelligence (Health, Efficiency, Risk)"

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses finance data with strict governance:
        - No data fabrication
        - No forced assumptions
        - Full signal availability tracking
        """

        df = df.copy()

        # -----------------------------
        # Time Detection (Governed)
        # -----------------------------
        self.time_context = build_time_context(df)
        self.time_col = self.time_context.time_column

        # -----------------------------
        # Signal Resolution (Soft)
        # -----------------------------
        self.cols = {
            # Corporate P&L
            "revenue": resolve_column(df, ["revenue", "sales", "turnover"]),
            "expense": resolve_column(df, ["expense", "cost", "opex", "cogs"]),
            "profit": resolve_column(df, ["profit", "net_income", "ebit", "ebitda"]),

            # Balance Sheet
            "assets": resolve_column(df, ["assets", "total_assets"]),
            "equity": resolve_column(df, ["equity", "shareholder_equity"]),
            "debt": resolve_column(df, ["debt", "liabilities", "total_debt"]),

            # Banking / Credit (Optional)
            "receivables": resolve_column(df, ["accounts_receivable", "receivables"]),
            "loans": resolve_column(df, ["loan_amount", "loans"]),
            "npa": resolve_column(df, ["non_performing_assets", "npa"]),
            "collateral": resolve_column(df, ["collateral_value"]),
            "interest": resolve_column(df, ["interest_expense", "interest"]),

            # Market / Price (Optional)
            "close": resolve_column(df, ["close", "adj_close", "price"]),
            "volume": resolve_column(df, ["volume"]),
        }

        # -----------------------------
        # Signal Availability Registry
        # -----------------------------
        self.available_signals = {
            k: v for k, v in self.cols.items() if v is not None
        }

        # -----------------------------
        # Numeric Safety (No Fabrication)
        # -----------------------------
        for col in self.available_signals.values():
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[,$]", "", regex=True)
                    .replace("", np.nan)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -----------------------------
        # Time Ordering (If Possible)
        # -----------------------------
        if self.time_col:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col], errors="coerce"
            )
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------
    # KPI Engine (Governed, Capability-Driven)
    # =====================================================
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes finance KPIs with strict governance:
        - No thresholds
        - No targets
        - No fabricated KPIs
        - Graceful degradation
        """
    
        kpis: Dict[str, Any] = {}
        c = self.available_signals
        time_col = self.time_col
    
        # =================================================
        # Sub-domain 1: Revenue Quality
        # =================================================
        if "revenue" in c:
            revenue_series = df[c["revenue"]].dropna()
            if not revenue_series.empty:
                kpis["revenue_total"] = revenue_series.sum()
                kpis["revenue_mean"] = revenue_series.mean()
                kpis["revenue_variability"] = revenue_series.std()
    
        # =================================================
        # Sub-domain 2: Cost Structure
        # =================================================
        if "expense" in c:
            expense_series = df[c["expense"]].dropna()
            if not expense_series.empty:
                kpis["expense_total"] = expense_series.sum()
                kpis["expense_mean"] = expense_series.mean()
                kpis["expense_variability"] = expense_series.std()
    
        # =================================================
        # Sub-domain 3: Profitability
        # =================================================
        if "profit" in c:
            profit_series = df[c["profit"]].dropna()
            if not profit_series.empty:
                kpis["profit_total"] = profit_series.sum()
                kpis["profit_mean"] = profit_series.mean()
                kpis["profit_variability"] = profit_series.std()
    
            if "revenue" in c:
                rev = df[c["revenue"]].mean()
                prof = df[c["profit"]].mean()
                if pd.notna(rev) and rev != 0:
                    kpis["profit_to_revenue_ratio"] = safe_divide(prof, rev)
    
        # =================================================
        # Sub-domain 4: Liquidity & Cash Proxies
        # =================================================
        if "receivables" in c and "revenue" in c:
            recv = df[c["receivables"]].mean()
            rev = df[c["revenue"]].mean()
            if pd.notna(recv) and pd.notna(rev):
                kpis["receivables_to_revenue_ratio"] = safe_divide(recv, rev)
    
        if "assets" in c and "debt" in c:
            assets = df[c["assets"]].mean()
            debt = df[c["debt"]].mean()
            if pd.notna(assets):
                kpis["debt_to_assets_ratio"] = safe_divide(debt, assets)
    
        # =================================================
        # Sub-domain 5: Financial Risk & Stability
        # =================================================
        if "revenue" in c:
            rev_series = df[c["revenue"]].dropna()
            if len(rev_series) > 2:
                kpis["revenue_stability"] = rev_series.std()
    
        if "expense" in c:
            exp_series = df[c["expense"]].dropna()
            if len(exp_series) > 2:
                kpis["expense_stability"] = exp_series.std()
    
        # =================================================
        # Sub-domain 6: Banking Health (Optional)
        # =================================================
        if "npa" in c and "loans" in c:
            npa = df[c["npa"]].mean()
            loans = df[c["loans"]].mean()
            if pd.notna(loans):
                kpis["npa_to_loans_ratio"] = safe_divide(npa, loans)
    
        if "collateral" in c and "loans" in c:
            coll = df[c["collateral"]].mean()
            loans = df[c["loans"]].mean()
            if pd.notna(coll):
                kpis["loan_to_collateral_ratio"] = safe_divide(loans, coll)
    
        # =================================================
        # Sub-domain 7: Market Variability (Optional)
        # =================================================
        if "close" in c and time_col:
            price_series = (
                df[[time_col, c["close"]]]
                .dropna()
                .sort_values(time_col)[c["close"]]
            )
    
            if len(price_series) > 3:
                returns = price_series.pct_change().dropna()
                if not returns.empty:
                    kpis["price_return_variability"] = returns.std()
                    kpis["price_return_mean"] = returns.mean()
    
        # =================================================
        # Sub-domain 8: Anomaly / Integrity Proxy
        # =================================================
        if "expense" in c:
            kpis["expense_benford_deviation"] = benford_deviation(df[c["expense"]])
    
        return kpis

    # =====================================================
    # Finance Domain — Block 4
    # Visual Engine (Governed, Executive-Safe)
    # =====================================================
    
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Generates finance visuals with strict governance:
        - No thresholds
        - No targets
        - No annualization
        - Observational only
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.available_signals
        time_col = self.time_col
    
        # ---------------------------------
        # Internal save helper
        # ---------------------------------
        def save(fig, name, caption, importance, category):
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": importance,
                "category": category
            })
    
        # ---------------------------------
        # Formatter
        # ---------------------------------
        def human_fmt(x, _):
            if pd.isna(x):
                return ""
            if abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
    
        # =================================================
        # Sub-domain: Market Variability
        # =================================================
        if "close" in c and time_col:
            series = (
                df[[time_col, c["close"]]]
                .dropna()
                .sort_values(time_col)
                .set_index(time_col)[c["close"]]
            )
    
            if len(series) > 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                series.plot(ax=ax)
                ax.set_title("Price Movement Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "price_trend.png",
                    "Observed price movement over time",
                    0.9,
                    "market"
                )
    
        # =================================================
        # Sub-domain: Revenue & Cost Structure
        # =================================================
        if "revenue" in c and "expense" in c:
            rev = df[c["revenue"]].dropna().sum()
            exp = df[c["expense"]].dropna().sum()
    
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Revenue", "Expense"],
                [rev, exp]
            )
            ax.set_title("Revenue and Expense Magnitudes")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "revenue_expense.png",
                "Relative magnitude of revenue and expenses",
                0.85,
                "corporate"
            )
    
        # =================================================
        # Sub-domain: Profitability Signal
        # =================================================
        if "profit" in c:
            prof_series = df[c["profit"]].dropna()
            if not prof_series.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                prof_series.plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Distribution of Profit Values")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "profit_distribution.png",
                    "Observed distribution of profit values",
                    0.8,
                    "corporate"
                )
    
        # =================================================
        # Sub-domain: Capital Structure
        # =================================================
        if "debt" in c and "equity" in c:
            debt = df[c["debt"]].dropna().mean()
            equity = df[c["equity"]].dropna().mean()
    
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Debt", "Equity"],
                [debt, equity]
            )
            ax.set_title("Average Capital Components")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "capital_structure.png",
                "Observed capital structure components",
                0.82,
                "risk"
            )
    
        # =================================================
        # Sub-domain: Banking Asset Composition (Optional)
        # =================================================
        if "loans" in c and "npa" in c:
            loans = df[c["loans"]].dropna().mean()
            npa = df[c["npa"]].dropna().mean()
    
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Loans", "Non-Performing Assets"],
                [loans, npa]
            )
            ax.set_title("Loan and Non-Performing Asset Levels")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "loan_npa_levels.png",
                "Observed loan and non-performing asset levels",
                0.8,
                "banking"
            )
    
        # =================================================
        # Sub-domain: Expense Integrity Proxy
        # =================================================
        if "expense" in c:
            deviation = benford_deviation(df[c["expense"]])
            if deviation > 0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(["Deviation"], [deviation])
                ax.set_title("Expense Digit Distribution Deviation")
                save(
                    fig,
                    "expense_benford.png",
                    "Observed deviation in leading digit distribution",
                    0.75,
                    "integrity"
                )
    
        # =================================================
        # Trim: Many → Few
        # =================================================
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]


    # =====================================================
    # Finance Domain — Block 5
    # Insight Engine (Composite-First, Executive-Safe)
    # =====================================================
    
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates composite financial insights without
        thresholds, targets, or judgment language.
        """
    
        insights: List[Dict[str, Any]] = []
    
        # -------------------------------------------------
        # Sub-domain: Revenue & Cost Structure
        # -------------------------------------------------
        if "revenue_total" in kpis and "expense_total" in kpis:
            insights.append({
                "type": "composite",
                "title": "Revenue and Expense Co-movement",
                "so_what": (
                    "Revenue and expense magnitudes are both present, "
                    "indicating cost structure evolves alongside revenue scale."
                )
            })
    
        if "revenue_variability" in kpis and "expense_variability" in kpis:
            insights.append({
                "type": "composite",
                "title": "Revenue vs Expense Variability",
                "so_what": (
                    "Revenue and expenses exhibit different levels of variability, "
                    "suggesting cost flexibility may differ from revenue movement."
                )
            })
    
        # -------------------------------------------------
        # Sub-domain: Profitability
        # -------------------------------------------------
        if "profit_mean" in kpis and "profit_variability" in kpis:
            insights.append({
                "type": "composite",
                "title": "Profit Level and Variability",
                "so_what": (
                    "Observed profit levels coexist with measurable variability, "
                    "indicating earnings stability may fluctuate over time."
                )
            })
    
        if "profit_to_revenue_ratio" in kpis:
            insights.append({
                "type": "composite",
                "title": "Profit Relative to Revenue",
                "so_what": (
                    "Profit represents a consistent proportion of revenue on average, "
                    "providing a structural view of profitability."
                )
            })
    
        # -------------------------------------------------
        # Sub-domain: Liquidity & Cash Proxies
        # -------------------------------------------------
        if "receivables_to_revenue_ratio" in kpis:
            insights.append({
                "type": "composite",
                "title": "Revenue Conversion to Receivables",
                "so_what": (
                    "A portion of revenue is reflected in receivables, "
                    "highlighting how sales translate into near-term cash positions."
                )
            })
    
        if "debt_to_assets_ratio" in kpis:
            insights.append({
                "type": "composite",
                "title": "Debt Relative to Asset Base",
                "so_what": (
                    "Debt levels are observed relative to total assets, "
                    "providing context on capital structure composition."
                )
            })
    
        # -------------------------------------------------
        # Sub-domain: Financial Stability
        # -------------------------------------------------
        if "revenue_stability" in kpis and "expense_stability" in kpis:
            insights.append({
                "type": "composite",
                "title": "Revenue and Expense Stability Comparison",
                "so_what": (
                    "Revenue and expense stability differ, which may influence "
                    "earnings predictability over time."
                )
            })
    
        # -------------------------------------------------
        # Sub-domain: Banking / Credit (Optional)
        # -------------------------------------------------
        if "npa_to_loans_ratio" in kpis:
            insights.append({
                "type": "composite",
                "title": "Loan Quality Composition",
                "so_what": (
                    "A portion of loan exposure is associated with non-performing assets, "
                    "providing insight into portfolio composition."
                )
            })
    
        # -------------------------------------------------
        # Sub-domain: Integrity / Anomaly Proxy
        # -------------------------------------------------
        if "expense_benford_deviation" in kpis and kpis["expense_benford_deviation"] > 0:
            insights.append({
                "type": "atomic",
                "title": "Expense Digit Pattern Deviation",
                "so_what": (
                    "Expense values show deviation from expected leading-digit patterns, "
                    "which may warrant contextual review alongside other signals."
                )
            })
    
        # -------------------------------------------------
        # Graceful Fallback
        # -------------------------------------------------
        if not insights:
            insights.append({
                "type": "atomic",
                "title": "Financial Signal Availability",
                "so_what": (
                    "Available financial signals were limited, "
                    "so insights are constrained to observable data presence."
                )
            })
    
        return insights

    # =====================================================
    # Finance Domain — Block 6
    # Recommendation Engine (Advisory, Executive-Safe)
    # =====================================================
    
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates finance recommendations with:
        - Advisory language only
        - No prescriptions
        - No urgency or priority labels
        - Clear linkage to observed insight themes
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        insight_titles = {
            i.get("title", "").lower()
            for i in self.generate_insights(df, kpis)
        }
    
        # -------------------------------------------------
        # Revenue & Cost Structure
        # -------------------------------------------------
        if any("revenue" in t and "expense" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "Review how revenue growth and expense expansion move together, "
                    "to better understand cost flexibility as scale changes."
                ),
                "related_area": "revenue_cost_structure"
            })
    
        if any("variability" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "Consider monitoring variability patterns across revenue and expenses "
                    "to improve planning confidence under changing conditions."
                ),
                "related_area": "financial_stability"
            })
    
        # -------------------------------------------------
        # Profitability
        # -------------------------------------------------
        if any("profit" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "Examine the drivers behind observed profit levels and fluctuations "
                    "to identify levers that influence earnings consistency."
                ),
                "related_area": "profitability"
            })
    
        if "profit relative to revenue" in insight_titles:
            recommendations.append({
                "recommendation": (
                    "Use the observed profit-to-revenue relationship as a reference "
                    "when evaluating pricing, mix, or cost structure decisions."
                ),
                "related_area": "profitability"
            })
    
        # -------------------------------------------------
        # Liquidity & Capital Structure
        # -------------------------------------------------
        if any("receivables" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "Review receivables patterns alongside revenue recognition "
                    "to better understand cash conversion dynamics."
                ),
                "related_area": "liquidity"
            })
    
        if any("debt" in t or "capital" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "Assess capital structure composition over time to understand "
                    "how funding choices align with asset utilization."
                ),
                "related_area": "capital_structure"
            })
    
        # -------------------------------------------------
        # Banking / Credit (Optional)
        # -------------------------------------------------
        if any("loan" in t or "asset" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "Monitor loan and asset composition trends to maintain visibility "
                    "into portfolio characteristics as conditions evolve."
                ),
                "related_area": "banking"
            })
    
        # -------------------------------------------------
        # Integrity / Data Quality
        # -------------------------------------------------
        if any("digit pattern" in t or "deviation" in t for t in insight_titles):
            recommendations.append({
                "recommendation": (
                    "If needed, review expense data collection and aggregation processes "
                    "to ensure consistency and contextual understanding of anomalies."
                ),
                "related_area": "data_integrity"
            })
    
        # -------------------------------------------------
        # Graceful Fallback
        # -------------------------------------------------
        if not recommendations:
            recommendations.append({
                "recommendation": (
                    "Continue monitoring available financial signals over time "
                    "to expand insight depth as data coverage improves."
                ),
                "related_area": "general"
            })
    
        return recommendations
# =====================================================
# Finance Domain — Block 7
# Domain Detector (Boundary-Safe)
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    """
    Boundary-safe detector for Finance domain.
    Detects financial datasets without colliding
    with retail, marketing, or operational domains.
    """

    domain_name = "finance"

    # Token clusters (soft signals, not requirements)
    PNL_TOKENS = {
        "revenue", "sales", "income",
        "expense", "cost", "cogs", "opex",
        "profit", "ebit", "ebitda"
    }

    BALANCE_SHEET_TOKENS = {
        "asset", "assets",
        "liability", "liabilities",
        "equity", "debt"
    }

    FINANCE_OPS_TOKENS = {
        "loan", "interest",
        "receivable", "payable",
        "ledger", "npa"
    }

    EXCLUDED_GENERIC_TOKENS = {
        "price", "volume", "date", "id"
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}

        pnl_hits = {c for c in cols if any(t in c for t in self.PNL_TOKENS)}
        bs_hits = {c for c in cols if any(t in c for t in self.BALANCE_SHEET_TOKENS)}
        ops_hits = {c for c in cols if any(t in c for t in self.FINANCE_OPS_TOKENS)}

        generic_hits = {c for c in cols if any(t in c for t in self.EXCLUDED_GENERIC_TOKENS)}

        # -------------------------------
        # Confidence Construction (Soft)
        # -------------------------------
        signal_groups_present = sum([
            bool(pnl_hits),
            bool(bs_hits),
            bool(ops_hits)
        ])

        # Base confidence grows with signal diversity
        confidence = min(signal_groups_present / 3.0, 1.0)

        # Penalize generic-only datasets
        if signal_groups_present == 1 and generic_hits:
            confidence *= 0.6

        # No finance signals at all
        if signal_groups_present == 0:
            confidence = 0.0

        return DomainDetectionResult(
            domain="finance",
            confidence=round(confidence, 2),
            metadata={
                "pnl_columns": sorted(pnl_hits),
                "balance_sheet_columns": sorted(bs_hits),
                "finance_ops_columns": sorted(ops_hits),
                "generic_columns": sorted(generic_hits)
            }
        )


# =====================================================
# Registration
# =====================================================

def register(registry):
    registry.register(
        name="finance",
        domain_cls=FinanceDomain,
        detector_cls=FinanceDomainDetector,
    )

