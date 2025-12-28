import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS (PURE, DEFENSIVE)
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["date", "timestamp", "period", "month", "year"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:10])
                return c
            except Exception:
                continue
    return None


def _benford_deviation(series: pd.Series) -> float:
    """Returns deviation score from Benford's Law."""
    s = series.dropna().astype(str)
    first_digits = s.str.lstrip("-").str.replace(".", "", regex=False).str[0]
    first_digits = first_digits[first_digits.str.isnumeric()]
    if first_digits.empty:
        return 0.0

    observed = first_digits.value_counts(normalize=True)
    benford = {str(d): np.log10(1 + 1/d) for d in range(1, 10)}
    deviation = sum(abs(observed.get(str(d), 0) - benford[str(d)]) for d in range(1, 9))
    return deviation


# =====================================================
# FINANCE DOMAIN — UNIVERSAL 10/10
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Universal Finance Intelligence (Corporate, Investment, Banking, Fraud)"

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        # Resolve columns
        revenue = resolve_column(df, "revenue") or resolve_column(df, "sales")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")
        profit = resolve_column(df, "profit") or resolve_column(df, "net_income")
        assets = resolve_column(df, "assets")
        equity = resolve_column(df, "equity")
        debt = resolve_column(df, "debt") or resolve_column(df, "liability")
        interest = resolve_column(df, "interest_expense")
        receivables = resolve_column(df, "accounts_receivable")
        loans = resolve_column(df, "loan_amount")
        npa = resolve_column(df, "non_performing_assets")
        collateral = resolve_column(df, "collateral_value")
        close = resolve_column(df, "close") or resolve_column(df, "price")

        # ---------- Profitability ----------
        if revenue:
            kpis["total_revenue"] = df[revenue].sum()
        if expense:
            kpis["total_expense"] = df[expense].sum()
        if profit:
            kpis["total_profit"] = df[profit].sum()
        elif "total_revenue" in kpis and "total_expense" in kpis:
            kpis["total_profit"] = kpis["total_revenue"] - kpis["total_expense"]

        if "total_profit" in kpis and "total_revenue" in kpis:
            kpis["gross_margin"] = _safe_div(kpis["total_profit"], kpis["total_revenue"])

        if assets and "total_profit" in kpis:
            kpis["roa"] = _safe_div(kpis["total_profit"], df[assets].mean())

        if equity and "total_profit" in kpis:
            kpis["roe"] = _safe_div(kpis["total_profit"], df[equity].mean())

        # ---------- Liquidity ----------
        if receivables and revenue:
            kpis["dso"] = _safe_div(df[receivables].mean(), df[revenue].mean()) * 365

        # ---------- Risk ----------
        if debt and equity:
            kpis["debt_to_equity"] = _safe_div(df[debt].mean(), df[equity].mean())

        if profit and interest:
            kpis["interest_coverage"] = _safe_div(df[profit].sum(), df[interest].sum())

        # ---------- Investment ----------
        if close and self.time_col:
            prices = df[close].dropna()
            returns = prices.pct_change().dropna()
            if not returns.empty:
                kpis["volatility"] = returns.std()
                kpis["sharpe_ratio"] = _safe_div(returns.mean(), returns.std())
                kpis["var_95"] = np.percentile(returns, 5)

        # ---------- Banking ----------
        if loans and collateral:
            kpis["ltv"] = _safe_div(df[loans].mean(), df[collateral].mean())

        if npa and loans:
            kpis["npa_rate"] = _safe_div(df[npa].sum(), df[loans].sum())

        # ---------- Fraud ----------
        if expense:
            kpis["benford_deviation"] = _benford_deviation(df[expense])

        return kpis

    # ---------------- VISUALS (10+ POSSIBLE, RANKED) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        kpis = self.calculate_kpis(df)
        time = self.time_col

        def save(fig, name, caption, importance):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": importance
            })

        # 1. Revenue vs Expense
        if "total_revenue" in kpis and "total_expense" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["Revenue", "Expense"], [kpis["total_revenue"], kpis["total_expense"]])
            ax.set_title("Revenue vs Expense")
            save(fig, "pnl.png", "Revenue vs Expense", 0.9)

        # 2. Profit Margin
        if "gross_margin" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["Margin"], [kpis["gross_margin"]])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.set_title("Profit Margin")
            save(fig, "margin.png", "Profit Margin", 0.8)

        # 3. DSO
        if "dso" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["DSO"], [kpis["dso"]])
            ax.set_title("Days Sales Outstanding")
            save(fig, "dso.png", "Receivables Efficiency", 0.85)

        # 4. Debt to Equity
        if "debt_to_equity" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["Debt/Equity"], [kpis["debt_to_equity"]])
            ax.set_title("Leverage Ratio")
            save(fig, "de_ratio.png", "Leverage Risk", 0.9)

        # 5. Sharpe Ratio
        if "sharpe_ratio" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["Sharpe"], [kpis["sharpe_ratio"]])
            ax.set_title("Risk-Adjusted Return")
            save(fig, "sharpe.png", "Investment Efficiency", 0.85)

        # 6. VaR
        if "var_95" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["VaR (95%)"], [kpis["var_95"]])
            ax.set_title("Value at Risk")
            save(fig, "var.png", "Downside Risk", 0.9)

        # 7. Benford
        if "benford_deviation" in kpis:
            fig, ax = plt.subplots()
            ax.bar(["Benford Deviation"], [kpis["benford_deviation"]])
            ax.set_title("Fraud Risk Indicator")
            save(fig, "benford.png", "Fraud Anomaly Score", 0.95)

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        if kpis.get("gross_margin", 1) < 0.1:
            insights.append({
                "level": "RISK",
                "title": "Shrinking Margins",
                "so_what": "Costs are rising faster than revenue."
            })

        if kpis.get("dso", 0) > 45:
            insights.append({
                "level": "WARNING",
                "title": "Slow Collections",
                "so_what": f"DSO is {kpis['dso']:.0f} days."
            })

        if kpis.get("debt_to_equity", 0) > 2:
            insights.append({
                "level": "RISK",
                "title": "High Leverage",
                "so_what": "Debt levels may threaten solvency."
            })

        if kpis.get("sharpe_ratio", 1) < 1:
            insights.append({
                "level": "WARNING",
                "title": "Poor Risk-Adjusted Returns",
                "so_what": "Returns do not justify risk taken."
            })

        if kpis.get("benford_deviation", 0) > 0.15:
            insights.append({
                "level": "RISK",
                "title": "Fraud Signal Detected",
                "so_what": "Transaction patterns deviate from Benford’s Law."
            })

        return insights or [{
            "level": "INFO",
            "title": "Financials Stable",
            "so_what": "No material financial risks detected."
        }]

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if kpis.get("gross_margin", 1) < 0.1:
            recs.append({"action": "Audit COGS and supplier pricing", "priority": "HIGH"})

        if kpis.get("dso", 0) > 45:
            recs.append({"action": "Automate invoice reminders and collections", "priority": "HIGH"})

        if kpis.get("debt_to_equity", 0) > 2:
            recs.append({"action": "Pause new debt and reduce leverage", "priority": "HIGH"})

        if kpis.get("sharpe_ratio", 1) < 1:
            recs.append({"action": "Rebalance portfolio toward lower volatility assets", "priority": "MEDIUM"})

        if kpis.get("benford_deviation", 0) > 0.15:
            recs.append({"action": "Trigger forensic audit on expense transactions", "priority": "CRITICAL"})

        return recs or [{"action": "Continue monitoring financial KPIs", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"
    TOKENS: Set[str] = {
        "revenue", "expense", "profit", "asset", "liability",
        "equity", "loan", "interest", "price", "volume", "budget"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.TOKENS)]
        confidence = min(len(hits) / 3, 1.0)
        return DomainDetectionResult(
            domain="finance",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


def register(registry):
    registry.register(
        name="finance",
        domain_cls=FinanceDomain,
        detector_cls=FinanceDomainDetector,
    )
