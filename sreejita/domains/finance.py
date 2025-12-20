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
    """Safely divides n by d."""
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Finance-safe time detector.
    Strategy:
    1. Check explicit names (Date, Year, Period).
    2. Check fuzzy matches (any column with 'date' in name).
    3. Fallback: Check ALL object/string columns for parsable dates.
    """
    
    # 1. Explicit & Fuzzy Candidates
    explicit = ["date", "timestamp", "time", "day", "period", "fiscal_date", "month", "year"]
    cols_lower = {c.lower(): c for c in df.columns}

    # Pass 1: Explicit Names
    for key in explicit:
        if key in cols_lower:
            col = cols_lower[key]
            if not df[col].isna().all():
                try:
                    pd.to_datetime(df[col].dropna().iloc[:10], errors="raise")
                    return col
                except Exception:
                    pass

    # Pass 2: Fuzzy Names (contains "date")
    for low, real in cols_lower.items():
        if "date" in low and not df[real].isna().all():
            try:
                pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                return real
            except Exception:
                pass

    # Pass 3: Brute Force (Object columns only)
    for col in df.select_dtypes(include=["object", "string", "datetime"]).columns:
        try:
            pd.to_datetime(df[col].dropna().iloc[:10], errors="raise")
            return col
        except Exception:
            continue

    return None


def _prepare_time_series(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Ensures time column is datetime type and sorted."""
    df_out = df.copy()
    try:
        df_out[time_col] = pd.to_datetime(df_out[time_col], errors="coerce")
        df_out = df_out.dropna(subset=[time_col])
        df_out = df_out.sort_values(time_col)
    except Exception:
        pass
    return df_out


# =====================================================
# FINANCE DOMAIN (UNIFIED: MARKET + CORPORATE)
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Financial Market & Corporate Finance Analytics"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Finance data needs OHLCV (Market) OR Revenue/Expense (Corporate).
        """
        # 1. Stock Market (OHLCV)
        has_market = any(
            resolve_column(df, c) is not None 
            for c in ["close", "adj_close", "price", "volume"]
        )
        
        # 2. Corporate Finance
        has_corp = any(
            resolve_column(df, c) is not None
            for c in ["revenue", "profit", "ebitda", "net_income", "asset", "liability", "expense"]
        )

        return has_market or has_corp

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

        # 1. Market Data (OHLCV)
        close_col = resolve_column(df, "close") or resolve_column(df, "adj_close")
        volume_col = resolve_column(df, "volume")
        
        if close_col and pd.api.types.is_numeric_dtype(df[close_col]):
            prices = df[close_col].dropna()
            
            if len(prices) > 1:
                start = prices.iloc[0]
                end = prices.iloc[-1]
                
                # Total Return (Guard against div by zero)
                if start != 0:
                    kpis["total_return"] = (end - start) / start
                
                kpis["current_price"] = end
                
                # Volatility (Daily Standard Deviation)
                daily_returns = prices.pct_change().dropna()
                kpis["volatility"] = daily_returns.std()
                
                # Max Drawdown (Pro Grade)
                running_max = prices.cummax()
                drawdowns = (prices - running_max) / running_max
                kpis["max_drawdown"] = drawdowns.min()

        if volume_col and pd.api.types.is_numeric_dtype(df[volume_col]):
            kpis["avg_volume"] = df[volume_col].mean()

        # 2. Corporate Finance
        revenue = resolve_column(df, "revenue") or resolve_column(df, "sales")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")
        profit = resolve_column(df, "profit") or resolve_column(df, "net_income")
        budget = resolve_column(df, "budget") or resolve_column(df, "target")
        
        if revenue and pd.api.types.is_numeric_dtype(df[revenue]):
            kpis["total_revenue"] = df[revenue].sum()
        
        if expense and pd.api.types.is_numeric_dtype(df[expense]):
            kpis["total_expense"] = df[expense].sum()

        if profit and pd.api.types.is_numeric_dtype(df[profit]):
            kpis["total_profit"] = df[profit].sum()
        elif "total_revenue" in kpis and "total_expense" in kpis:
            kpis["total_profit"] = kpis["total_revenue"] - kpis["total_expense"]

        if "total_profit" in kpis and "total_revenue" in kpis:
            kpis["profit_margin"] = _safe_div(kpis["total_profit"], kpis["total_revenue"])

        # 3. Budget Logic
        if budget and revenue and pd.api.types.is_numeric_dtype(df[budget]):
            actual = df[revenue].sum()
            planned = df[budget].sum()
            if planned != 0:
                var = actual - planned
                kpis["budget_variance_pct"] = _safe_div(var, planned)

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate KPIs once
        kpis = self.calculate_kpis(df)

        close_col = resolve_column(df, "close") or resolve_column(df, "adj_close")
        volume_col = resolve_column(df, "volume")
        revenue = resolve_column(df, "revenue") or resolve_column(df, "sales")
        expense = resolve_column(df, "expense") or resolve_column(df, "cost")

        def human_fmt(x, _):
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        # -------- Visual 1: Price Trend (Market) --------
        if self.has_time_series and close_col and pd.api.types.is_numeric_dtype(df[close_col]):
            p = output_dir / "price_trend.png"
            plt.figure(figsize=(7, 4))
            
            # Use aggregated plot if data is huge
            plot_df = df
            if len(df) > 300:
                plot_df = df.iloc[::max(1, len(df)//200)]

            plt.plot(plot_df[self.time_col], plot_df[close_col], linewidth=2, color="#1f77b4")
            plt.title("Price History")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Asset price movement over time"})

        # -------- Visual 2: Revenue vs Expense (Corporate) --------
        if revenue and expense and pd.api.types.is_numeric_dtype(df[revenue]) and pd.api.types.is_numeric_dtype(df[expense]):
            p = output_dir / "pnl_summary.png"
            total_rev = df[revenue].sum()
            total_exp = df[expense].sum()
            
            plt.figure(figsize=(7, 4))
            bars = plt.bar(["Revenue", "Expenses"], [total_rev, total_exp], color=["#2ca02c", "#d62728"])
            
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                         human_fmt(bar.get_height(), None), ha='center', va='bottom')
                         
            plt.title("Revenue vs Expenses")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "P&L Overview"})

        # -------- Visual 3: Volume / Ratios --------
        if self.has_time_series and volume_col and pd.api.types.is_numeric_dtype(df[volume_col]):
            p = output_dir / "volume_trend.png"
            plt.figure(figsize=(7, 4))
            plot_df = df if len(df) < 100 else df.iloc[::max(1, len(df)//100)]
            plt.bar(plot_df[self.time_col], plot_df[volume_col], color="#7f7f7f", alpha=0.6)
            plt.title("Trading Volume")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Trading activity volume"})
        
        elif "profit_margin" in kpis:
            # Show Financial Ratios if no market volume
            p = output_dir / "ratios.png"
            ratios = {k: v for k, v in kpis.items() if "pct" in k or "margin" in k}
            if ratios:
                plt.figure(figsize=(7, 4))
                plt.bar(ratios.keys(), ratios.values(), color="#1f77b4")
                plt.title("Key Financial Ratios")
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
                plt.tight_layout()
                plt.savefig(p)
                plt.close()
                visuals.append({"path": p, "caption": "Key efficiency metrics"})

        return visuals[:4]

    # ---------------- ATOMIC INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        ret = kpis.get("total_return")
        vol = kpis.get("volatility")
        drawdown = kpis.get("max_drawdown")
        margin = kpis.get("profit_margin")
        var_pct = kpis.get("budget_variance_pct")
        
        # 1. Market Insights
        if ret is not None:
            sentiment = "Positive" if ret > 0 else "Negative"
            insights.append({
                "level": "INFO",
                "title": f"Market Trend: {sentiment}",
                "so_what": f"Total return over the period is {ret:.2%}."
            })
            
        if vol is not None and 0.02 < vol < 0.20:
            insights.append({
                "level": "RISK",
                "title": "High Volatility",
                "so_what": f"Daily fluctuation is {vol:.2%}. Asset is risky."
            })
            
        if drawdown is not None:
            if drawdown < -0.30:
                insights.append({
                    "level": "RISK",
                    "title": "Severe Drawdown Detected",
                    "so_what": f"The asset experienced a maximum drawdown of {abs(drawdown):.1%}, indicating high downside risk."
                })
            elif drawdown < -0.15:
                insights.append({
                    "level": "WARNING",
                    "title": "Moderate Drawdown Observed",
                    "so_what": f"Maximum drawdown reached {abs(drawdown):.1%}. Risk management may be required."
                })

        # 2. Corporate Insights
        if margin is not None:
            if margin < 0:
                insights.append({
                    "level": "RISK",
                    "title": "Negative Profit Margin",
                    "so_what": f"Net margin is {margin:.1%}. Costs exceed revenue."
                })
            elif margin > 0.15:
                insights.append({
                    "level": "INFO",
                    "title": "Healthy Profitability",
                    "so_what": f"Net margin is strong at {margin:.1%}."
                })

        # 3. Budget Insights
        if var_pct is not None:
            if var_pct < -0.05:
                insights.append({
                    "level": "RISK",
                    "title": "Missed Budget Target",
                    "so_what": f"Revenue is {abs(var_pct):.1%} below plan."
                })

        # Call Composite Logic
        insights += self.generate_composite_insights(df, kpis)

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Finance Metrics Detected",
                "so_what": "Financial data is available for analysis."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (v3.0 INTELLIGENCE) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Multi-KPI reasoning layer.
        Combines signals to produce analyst-grade insights.
        """
        insights: List[Dict[str, Any]] = []

        # Market KPIs
        ret = kpis.get("total_return")
        vol = kpis.get("volatility")
        dd = kpis.get("max_drawdown")
        avg_vol = kpis.get("avg_volume")

        # Corporate KPIs
        margin = kpis.get("profit_margin")
        budget_var = kpis.get("budget_variance_pct")

        # 1. Strong Returns + Deep Drawdown + Low Volatility
        if ret is not None and vol is not None and dd is not None:
            if ret > 0.25 and dd < -0.30 and vol < 0.02:
                insights.append({
                    "level": "RISK",
                    "title": "Event-Driven Downside Risk",
                    "so_what": (
                        f"Despite strong returns ({ret:.1%}), the asset experienced "
                        f"a severe drawdown ({abs(dd):.1%}) while volatility remained low "
                        f"({vol:.1%}). This suggests sharp, event-driven corrections rather "
                        f"than a failing trend."
                    )
                })

        # 2. Healthy Trend Confirmation
        if ret is not None and dd is not None:
            if ret > 0.15 and dd > -0.15:
                insights.append({
                    "level": "INFO",
                    "title": "Stable Uptrend with Controlled Risk",
                    "so_what": (
                        f"The asset shows solid growth ({ret:.1%}) with limited downside "
                        f"risk (max drawdown {abs(dd):.1%}), indicating a healthy trend."
                    )
                })

        # 3. Liquidity Risk During Drawdowns
        if dd is not None and avg_vol is not None:
            # Simple heuristic: if volume is very low compared to history length
            if dd < -0.30 and avg_vol < df.shape[0] * 10:
                insights.append({
                    "level": "WARNING",
                    "title": "Potential Liquidity-Driven Sell-Off",
                    "so_what": (
                        f"A deep drawdown ({abs(dd):.1%}) combined with relatively low "
                        f"trading volume suggests liquidity constraints may amplify losses "
                        f"during market stress."
                    )
                })

        # 4. Corporate: Profitable but Strategically Risky
        if margin is not None and budget_var is not None:
            if margin > 0.15 and budget_var < -0.05:
                insights.append({
                    "level": "WARNING",
                    "title": "Profitable but Missing Growth Targets",
                    "so_what": (
                        f"Profit margins are healthy ({margin:.1%}), but revenue is "
                        f"below budget ({abs(budget_var):.1%}). This may indicate "
                        f"underinvestment or slowing growth."
                    )
                })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        # Market Recs
        if kpis.get("volatility", 0) > 0.02 or kpis.get("max_drawdown", 0) < -0.20:
            recs.append({
                "action": "Review risk exposure and stop-loss levels",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
            
        # Corporate Recs
        if kpis.get("profit_margin", 1) < 0:
            recs.append({
                "action": "Audit top expense categories to reduce burn rate",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
            
        if not recs:
            recs.append({
                "action": "Continue monitoring financial performance",
                "priority": "LOW",
                "timeline": "Ongoing"
            })
            
        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    domain_name = "finance"

    FINANCE_TOKENS: Set[str] = {
        # Market Data
        "open", "close", "high", "low", "volume", "adj_close", "ticker",
        "portfolio", "asset", "equity", "volatility",
        
        # Corporate
        "revenue", "income", "sales",
        "expense", "cost", "spend",
        "budget", "forecast",
        "profit", "loss", "net_income",
        "ledger", "fiscal", "amount", "balance_sheet", "cash_flow", "liability"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.FINANCE_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)
        
        # OHLC Dominance Rule (Market Data)
        ohlc_exclusive = all(
            any(t in c for c in cols) 
            for t in ["open", "high", "low", "close"]
        )
        
        if ohlc_exclusive:
            confidence = 0.95

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

