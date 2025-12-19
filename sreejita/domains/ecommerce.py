import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS
# =====================================================

def _safe_div(n, d):
    """Safely divides n by d."""
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    E-com-safe time detector:
    session time, order time, event time.
    """
    candidates = [
        "session_date", "visit_date", "event_time", "timestamp",
        "order_date", "transaction_date", "date"
    ]

    cols = {c.lower(): c for c in df.columns}

    for key in candidates:
        for low, real in cols.items():
            if key in low and not df[real].isna().all():
                try:
                    pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                    return real
                except Exception:
                    continue
    return None


# =====================================================
# E-COMMERCE DOMAIN
# =====================================================

class EcommerceDomain(BaseDomain):
    name = "ecommerce"
    description = "E-Commerce Analytics (Traffic, Conversion, Cart Abandonment)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        E-com data needs web metrics (Sessions, Pageviews) OR sales funnel data.
        """
        return any(
            resolve_column(df, c) is not None
            for c in [
                "session_id", "visit_id", "visitor",
                "pageviews", "bounce_rate", "traffic_source",
                "add_to_cart", "checkout", "conversion",
                "order_id", "transaction_id" # Overlaps with Retail, but context differs
            ]
        )

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        self.has_time_series = False

        if self.time_col:
            df = df.copy()
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.dropna(subset=[self.time_col])
            df = df.sort_values(self.time_col)
            self.has_time_series = df[self.time_col].nunique() >= 2

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        # Traffic Metrics
        sessions = resolve_column(df, "sessions") or resolve_column(df, "visits")
        users = resolve_column(df, "users") or resolve_column(df, "visitors")
        pageviews = resolve_column(df, "pageviews")
        
        # Funnel Metrics
        orders = resolve_column(df, "orders") or resolve_column(df, "transactions")
        revenue = resolve_column(df, "revenue") or resolve_column(df, "sales")
        add_to_cart = resolve_column(df, "add_to_cart") or resolve_column(df, "atc")
        
        # 1. Traffic Volume
        if sessions and pd.api.types.is_numeric_dtype(df[sessions]):
            kpis["total_sessions"] = df[sessions].sum()
        elif users:
            kpis["total_users"] = df[users].nunique() if not pd.api.types.is_numeric_dtype(df[users]) else df[users].sum()

        # 2. Conversion Rate (CR)
        # Formula: Orders / Sessions (if both exist)
        if orders and sessions and pd.api.types.is_numeric_dtype(df[orders]) and pd.api.types.is_numeric_dtype(df[sessions]):
            total_orders = df[orders].sum()
            total_sessions = df[sessions].sum()
            kpis["conversion_rate"] = _safe_div(total_orders, total_sessions)
            kpis["target_conversion_rate"] = 0.025 # 2.5% Benchmark

        # 3. Cart Abandonment Rate
        # Formula: 1 - (Orders / Add To Carts)
        if orders and add_to_cart and pd.api.types.is_numeric_dtype(df[orders]) and pd.api.types.is_numeric_dtype(df[add_to_cart]):
            total_orders = df[orders].sum()
            total_atc = df[add_to_cart].sum()
            
            if total_atc > 0:
                kpis["cart_abandonment_rate"] = 1.0 - (total_orders / total_atc)
                kpis["target_abandonment_rate"] = 0.70 # < 70% is good

        # 4. Average Order Value (AOV)
        if revenue and orders and pd.api.types.is_numeric_dtype(df[revenue]) and pd.api.types.is_numeric_dtype(df[orders]):
            kpis["aov"] = _safe_div(df[revenue].sum(), df[orders].sum())

        return kpis

    # ---------------- VISUALS (MAX 4) ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        kpis = self.calculate_kpis(df)

        sessions = resolve_column(df, "sessions") or resolve_column(df, "visits")
        orders = resolve_column(df, "orders") or resolve_column(df, "transactions")
        add_to_cart = resolve_column(df, "add_to_cart") or resolve_column(df, "atc")
        source = resolve_column(df, "source") or resolve_column(df, "channel") or resolve_column(df, "medium")
        device = resolve_column(df, "device") or resolve_column(df, "platform")

        def human_fmt(x, _):
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        # -------- Visual 1: Traffic Trend --------
        if self.has_time_series and sessions and pd.api.types.is_numeric_dtype(df[sessions]):
            p = output_dir / "traffic_trend.png"
            plt.figure(figsize=(7, 4))

            plot_df = df.copy()
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME")
                    .sum()
                    .reset_index()
                )

            plt.plot(plot_df[self.time_col], plot_df[sessions], linewidth=2, color="#17becf")
            plt.title("Web Traffic Trend (Sessions)")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Visitor traffic over time"})

        # -------- Visual 2: Conversion Funnel (Bar) --------
        if sessions and add_to_cart and orders:
            # Check numeric types
            cols = [sessions, add_to_cart, orders]
            if all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
                p = output_dir / "conversion_funnel.png"
                
                funnel_data = {
                    "Sessions": df[sessions].sum(),
                    "Add to Cart": df[add_to_cart].sum(),
                    "Purchases": df[orders].sum()
                }
                
                plt.figure(figsize=(7, 4))
                plt.bar(funnel_data.keys(), funnel_data.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
                plt.title("Conversion Funnel")
                plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
                
                # Add text labels
                for i, v in enumerate(funnel_data.values()):
                    plt.text(i, v, human_fmt(v, None), ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(p)
                plt.close()
                visuals.append({"path": p, "caption": "User journey drop-off points"})

        # -------- Visual 3: Traffic Source Breakdown --------
        metric = sessions or orders # Use sessions if avail, else orders
        if source and metric and pd.api.types.is_numeric_dtype(df[metric]):
            p = output_dir / "traffic_source.png"
            
            top_sources = df.groupby(source)[metric].sum().sort_values(ascending=False).head(7)
            
            plt.figure(figsize=(7, 4))
            top_sources.plot(kind="barh", color="#9467bd")
            plt.title(f"Top Traffic Sources by {metric.title()}")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Best performing marketing channels"})

        # -------- Visual 4: Device/Platform Mix --------
        if device and metric and pd.api.types.is_numeric_dtype(df[metric]):
            p = output_dir / "device_split.png"
            
            # Cap at top 5 to keep pie chart clean
            dev_counts = df.groupby(device)[metric].sum().sort_values(ascending=False).head(5)
            
            plt.figure(figsize=(6, 4))
            dev_counts.plot(kind="pie", autopct='%1.1f%%')
            plt.ylabel("")
            plt.title(f"Device Breakdown ({metric.title()})")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "User device preference"})

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        cr = kpis.get("conversion_rate")
        abandonment = kpis.get("cart_abandonment_rate")
        aov = kpis.get("aov")

        # 1. Conversion Rate
        if cr is not None:
            if cr < 0.01:
                insights.append({
                    "level": "RISK",
                    "title": "Low Conversion Rate",
                    "so_what": f"CR is {cr:.2%}, below the 1% critical threshold. Traffic is not buying."
                })
            elif cr < 0.025:
                insights.append({
                    "level": "WARNING",
                    "title": "Suboptimal Conversion",
                    "so_what": f"CR is {cr:.2%}, slightly below the 2.5% e-commerce benchmark."
                })

        # 2. Cart Abandonment
        if abandonment is not None:
            if abandonment > 0.80:
                insights.append({
                    "level": "RISK",
                    "title": "High Cart Abandonment",
                    "so_what": f"{abandonment:.1%} of carts are abandoned. Checkout friction likely exists."
                })
            elif abandonment > 0.70:
                insights.append({
                    "level": "WARNING",
                    "title": "Cart Abandonment Alert",
                    "so_what": f"Abandonment rate is {abandonment:.1%}, monitor checkout flow."
                })

        # 3. AOV
        if aov is not None:
             insights.append({
                "level": "INFO",
                "title": "Order Value Healthy",
                "so_what": f"Average Order Value (AOV) is holding at {int(aov)}."
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Store Performance Stable",
                "so_what": "Traffic and conversion metrics are within expected ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        cr = kpis.get("conversion_rate")
        abandonment = kpis.get("cart_abandonment_rate")

        if abandonment is not None and abandonment > 0.75:
            recs.append({
                "action": "Implement abandoned cart recovery emails",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
        
        if cr is not None and cr < 0.02:
            recs.append({
                "action": "Audit landing pages and checkout flow for friction",
                "priority": "MEDIUM",
                "timeline": "This Week"
            })

        if not recs:
            recs.append({
                "action": "Continue optimizing traffic sources",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR (COLLISION PROOF)
# =====================================================

class EcommerceDomainDetector(BaseDomainDetector):
    domain_name = "ecommerce"

    ECOM_TOKENS: Set[str] = {
        # Web Metrics
        "session", "visit", "pageview", "bounce_rate",
        "traffic_source", "medium", "campaign", "referrer",
        
        # Funnel
        "add_to_cart", "checkout", "cart_abandonment",
        "conversion_rate", "transaction", "aov",
        
        # Tech
        "device", "browser", "os", "platform"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        
        hits = [c for c in cols if any(t in c for t in self.ECOM_TOKENS)]
        
        # Base confidence
        confidence = min(len(hits) / 3, 1.0)

        # 🔑 E-COM DOMINANCE RULE
        # Distinguish from Retail: Retail focuses on "Store/Product", E-com focuses on "Session/Web"
        ecom_exclusive = any(
            t in c 
            for c in cols 
            for t in {"session", "pageview", "bounce", "traffic", "browser", "device"}
        )
        
        if ecom_exclusive:
            confidence = max(confidence, 0.85)

        return DomainDetectionResult(
            domain="ecommerce",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="ecommerce",
        domain_cls=EcommerceDomain,
        detector_cls=EcommerceDomainDetector,
    )
