import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d

def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["date", "timestamp", "period", "month", "year", "fiscal_date"]
    for c in df.columns:
        for k in candidates:
            if k in c.lower():
                try:
                    pd.to_datetime(df[c].dropna().iloc[0])
                    return c
                except Exception:
                    continue
    return None

def _benford_deviation(series: pd.Series) -> float:
    """Calculates deviation from Benford's Law (Fraud Detection)."""
    s = series.dropna().astype(str)
    first_digits = s.str.lstrip("-").str.replace(".", "", regex=False).str[0]
    first_digits = first_digits[first_digits.str.isnumeric()]
    
    if len(first_digits) < 100: return 0.0

    observed = first_digits.value_counts(normalize=True)
    benford = {str(d): np.log10(1 + 1/d) for d in range(1, 10)}
    
    deviation = sum(abs(observed.get(str(d), 0) - benford[str(d)]) for d in range(1, 10))
    return deviation


# =====================================================
# FINANCE DOMAIN (UNIVERSAL 10/10)
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Universal Finance Intelligence (Corporate, Market, Banking, Fraud)"

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        self.cols = {
            # Corporate
            "revenue": resolve_column(df, "revenue") or resolve_column(df, "sales"),
            "expense": resolve_column(df, "expense") or resolve_column(df, "cost") or resolve_column(df, "opex"),
            "profit": resolve_column(df, "profit") or resolve_column(df, "net_income") or resolve_column(df, "ebitda"),
            # Balance Sheet
            "assets": resolve_column(df, "assets"),
            "equity": resolve_column(df, "equity"),
            "debt": resolve_column(df, "debt") or resolve_column(df, "liabilities"),
            # Banking
            "receivables": resolve_column(df, "accounts_receivable"),
            "loans": resolve_column(df, "loan_amount"),
            "npa": resolve_column(df, "non_performing_assets"),
            "collateral": resolve_column(df, "collateral_value"),
            "interest": resolve_column(df, "interest_expense"),
            # Market
            "close": resolve_column(df, "close") or resolve_column(df, "adj_close") or resolve_column(df, "price"),
            "volume": resolve_column(df, "volume"),
        }
        
        # Numeric Safety
        for c in self.cols.values():
            if c and df[c].dtype == object:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)

        # Date Cleaning
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Profitability
        if c["revenue"]: kpis["total_revenue"] = df[c["revenue"]].sum()
        if c["expense"]: kpis["total_expense"] = df[c["expense"]].sum()
        
        if c["profit"]:
            kpis["total_profit"] = df[c["profit"]].sum()
        elif c["revenue"] and c["expense"]:
            kpis["total_profit"] = kpis["total_revenue"] - kpis["total_expense"]

        if kpis.get("total_revenue") and kpis.get("total_profit"):
            kpis["gross_margin"] = _safe_div(kpis["total_profit"], kpis["total_revenue"])

        # 2. Solvency & Risk
        if c["debt"] and c["equity"]:
            kpis["debt_to_equity"] = _safe_div(df[c["debt"]].mean(), df[c["equity"]].mean())

        if c["profit"] and c["interest"]:
            # Interest Coverage Ratio
            kpis["interest_coverage"] = _safe_div(df[c["profit"]].sum(), df[c["interest"]].sum())

        # 3. Liquidity
        if c["receivables"] and c["revenue"]:
            # Days Sales Outstanding (DSO)
            kpis["dso"] = _safe_div(df[c["receivables"]].mean(), df[c["revenue"]].mean()) * 365

        # 4. Banking Health
        if c["npa"] and c["loans"]:
            kpis["npa_rate"] = _safe_div(df[c["npa"]].sum(), df[c["loans"]].sum())

        if c["loans"] and c["collateral"]:
            kpis["ltv"] = _safe_div(df[c["loans"]].mean(), df[c["collateral"]].mean())

        # 5. Market Stats
        if c["close"] and self.time_col:
            prices = df.set_index(self.time_col)[c["close"]].sort_index()
            returns = prices.pct_change().dropna()
            kpis["volatility"] = returns.std() * np.sqrt(252)
            if returns.std() > 0:
                kpis["sharpe_ratio"] = (returns.mean() * 252) / kpis["volatility"]
            kpis["var_95"] = np.percentile(returns, 5)

        # 6. Fraud Score
        if c["expense"]:
            kpis["benford_deviation"] = _benford_deviation(df[c["expense"]])

        return kpis

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        kpis = self.calculate_kpis(df)
        c = self.cols

        def save(fig, name, caption, imp, cat):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": imp,
                "category": cat
            })

        def human_fmt(x, _):
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        # 1. Market Price Trend
        if c["close"] and self.time_col:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(self.time_col)[c["close"]].plot(ax=ax, color="#1f77b4")
            ax.set_title("Asset Price Trend")
            save(fig, "price_trend.png", "Price history", 0.9, "market")

        # 2. Revenue vs Expense (Waterfall or Bar)
        if kpis.get("total_revenue") and kpis.get("total_expense"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Revenue", "Expense", "Profit"], 
                   [kpis["total_revenue"], kpis["total_expense"], kpis["total_profit"]], 
                   color=["green", "red", "blue"])
            ax.set_title("P&L Overview")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "pnl.png", "Profitability summary", 0.85, "corporate")

        # 3. Benford Fraud Detection
        if kpis.get("benford_deviation", 0) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            score = kpis["benford_deviation"]
            ax.bar(["Anomaly Score"], [score], color="red" if score > 0.15 else "orange")
            ax.axhline(0.15, color="black", linestyle="--", label="Fraud Threshold")
            ax.set_ylim(0, 0.3)
            ax.set_title("Fraud Detection (Benford's Law)")
            save(fig, "fraud.png", "Anomaly detection", 1.0 if score > 0.15 else 0.6, "fraud")

        # 4. Leverage Ratio
        if kpis.get("debt_to_equity"):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Debt/Equity"], [kpis["debt_to_equity"]], color="maroon")
            ax.axhline(2.0, color="gray", linestyle="--")
            ax.set_title("Leverage Ratio")
            save(fig, "leverage.png", "Solvency risk", 0.88, "risk")

        # 5. Volatility / Risk Profile
        if kpis.get("volatility"):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Volatility"], [kpis["volatility"]], color="purple")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.set_title("Annualized Volatility")
            save(fig, "volatility.png", "Market risk", 0.8, "risk")

        # 6. Interest Coverage (New!)
        if c["profit"] and c["interest"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["EBIT", "Interest"], [df[c["profit"]].sum(), df[c["interest"]].sum()])
            ax.set_title(f"Interest Coverage: {kpis.get('interest_coverage',0):.1f}x")
            save(fig, "coverage.png", "Debt service ability", 0.85, "corporate")

        # 7. NPA Composition (New!)
        if c["npa"] and c["loans"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            npa_val = df[c["npa"]].sum()
            total_val = df[c["loans"]].sum()
            ax.pie([total_val - npa_val, npa_val], labels=["Good Loans", "NPA"], autopct='%1.1f%%', colors=["#2ca02c", "#d62728"])
            ax.set_title("Loan Portfolio Health")
            save(fig, "npa_pie.png", "Asset quality", 0.92, "banking")

        # 8. Gross Margin Trend (New!)
        if self.time_col and c["revenue"] and c["profit"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            temp = df.set_index(self.time_col).resample('M').sum()
            margin = temp[c["profit"]] / temp[c["revenue"]].replace(0, 1)
            margin.plot(ax=ax, color="green")
            ax.set_title("Gross Margin Trend")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
            save(fig, "margin_trend.png", "Profitability trend", 0.8, "corporate")

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        margin = kpis.get("gross_margin", 1.0)
        debt_eq = kpis.get("debt_to_equity", 0)
        int_cov = kpis.get("interest_coverage", 10)
        dso = kpis.get("dso", 0)
        benford = kpis.get("benford_deviation", 0)

        # Composite 1: Leverage Trap (High Debt + Low Ability to Pay)
        if debt_eq > 2.0 and int_cov < 1.5:
            insights.append({
                "level": "CRITICAL", "title": "Leverage Trap",
                "so_what": f"High debt ({debt_eq:.1f}x equity) with dangerously low interest coverage ({int_cov:.1f}x)."
            })

        # Composite 2: Paper Profits (Profit exists, but Cash is stuck)
        if margin > 0.10 and dso > 90:
            insights.append({
                "level": "RISK", "title": "Paper Profits Warning",
                "so_what": f"Profitable on paper, but cash is stuck in receivables (DSO: {dso:.0f} days)."
            })

        # Atomic Fallbacks
        if benford > 0.15:
            insights.append({
                "level": "CRITICAL", "title": "Potential Fraud Detected",
                "so_what": "Expense values violate Benford's Law naturally occurring patterns."
            })

        if margin < 0.10 and not any("Profits" in i["title"] for i in insights):
            insights.append({
                "level": "WARNING", "title": "Thin Margins",
                "so_what": f"Gross margin is {margin:.1%}, leaving little room for opex."
            })

        if not insights:
            insights.append({"level": "INFO", "title": "Financials Stable", "so_what": "Solvency and profitability metrics healthy."})

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if "Leverage Trap" in titles:
            recs.append({"action": "Freeze new borrowing and restructure existing debt immediately.", "priority": "HIGH"})
        
        if "Paper Profits Warning" in titles:
            recs.append({"action": "Aggressively chase overdue invoices and tighten credit terms.", "priority": "HIGH"})

        if kpis.get("benford_deviation", 0) > 0.15:
            recs.append({"action": "Trigger forensic audit of expense ledger.", "priority": "CRITICAL"})

        if not recs:
            recs.append({"action": "Reinvest surplus cash into growth or debt reduction.", "priority": "LOW"})

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"
    TOKENS = {"revenue", "expense", "profit", "asset", "liability", "equity", "loan", "interest", "price", "volume", "ledger"}

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # Boost if Balance Sheet terms exist
        if any("asset" in c for c in hits) and any("liability" in c for c in hits):
            confidence = max(confidence, 0.95)
            
        return DomainDetectionResult("finance", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("finance", FinanceDomain, FinanceDomainDetector)
