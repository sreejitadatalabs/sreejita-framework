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
    """E-com specific time detector."""
    candidates = ["session_date", "visit_date", "timestamp", "order_date", "date"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except:
                continue
    return None


# =====================================================
# E-COMMERCE DOMAIN (UNIVERSAL 10/10)
# =====================================================

class EcommerceDomain(BaseDomain):
    name = "ecommerce"
    description = "Universal E-Commerce Analytics (Traffic, Conversion, Funnel, Retention)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        # 1. Resolve columns ONCE.
        self.cols = {
            # Traffic
            "sessions": resolve_column(df, "sessions") or resolve_column(df, "visits"),
            "users": resolve_column(df, "users") or resolve_column(df, "visitors"),
            "pageviews": resolve_column(df, "pageviews") or resolve_column(df, "screen_views"),
            "bounce": resolve_column(df, "bounce_rate"),
            
            # Funnel
            "add_to_cart": resolve_column(df, "add_to_cart") or resolve_column(df, "atc"),
            "checkout": resolve_column(df, "checkout") or resolve_column(df, "begin_checkout"),
            "orders": resolve_column(df, "orders") or resolve_column(df, "transactions"),
            "revenue": resolve_column(df, "revenue") or resolve_column(df, "sales"),
            "returns": resolve_column(df, "returns") or resolve_column(df, "refunds"),
            
            # Dimensions
            "source": resolve_column(df, "source") or resolve_column(df, "channel"),
            "device": resolve_column(df, "device") or resolve_column(df, "platform"),
            "product": resolve_column(df, "product_name") or resolve_column(df, "sku"),
            "category": resolve_column(df, "category")
        }

        # 2. Date Cleaning
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Traffic
        if c["sessions"]: kpis["total_sessions"] = df[c["sessions"]].sum()
        if c["users"]: kpis["total_users"] = df[c["users"]].sum()
        if c["bounce"]: kpis["avg_bounce_rate"] = df[c["bounce"]].mean()

        # 2. Conversion & Funnel
        if c["orders"] and c["sessions"]:
            kpis["conversion_rate"] = _safe_div(df[c["orders"]].sum(), df[c["sessions"]].sum())

        if c["add_to_cart"] and c["orders"]:
            atc = df[c["add_to_cart"]].sum()
            orders = df[c["orders"]].sum()
            if atc > 0:
                kpis["cart_abandonment_rate"] = max(0.0, 1.0 - (orders / atc))

        if c["checkout"] and c["orders"]:
            checkouts = df[c["checkout"]].sum()
            orders = df[c["orders"]].sum()
            if checkouts > 0:
                kpis["checkout_dropoff_rate"] = max(0.0, 1.0 - (orders / checkouts))

        # 3. Economics
        if c["revenue"] and c["orders"]:
            kpis["aov"] = _safe_div(df[c["revenue"]].sum(), df[c["orders"]].sum())

        if c["returns"] and c["orders"]:
            kpis["return_rate"] = _safe_div(df[c["returns"]].sum(), df[c["orders"]].sum())

        return kpis

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        kpis = self.calculate_kpis(df)

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
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        # 1. Traffic Trend (Sessions)
        if self.time_col and c["sessions"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(self.time_col).resample('M')[c["sessions"]].sum().plot(ax=ax, color="#17becf")
            ax.set_title("Traffic Trend (Sessions)")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "traffic_trend.png", "Visitor volume", 0.9, "traffic")

        # 2. Conversion Funnel (Critical)
        if c["sessions"] and c["add_to_cart"] and c["orders"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            funnel = [df[c["sessions"]].sum(), df[c["add_to_cart"]].sum(), df[c["orders"]].sum()]
            labels = ["Sessions", "Add to Cart", "Purchases"]
            ax.bar(labels, funnel, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_title("Conversion Funnel")
            
            imp = 0.95 if kpis.get("cart_abandonment_rate", 0) > 0.75 else 0.85
            save(fig, "funnel.png", "User journey drop-off", imp, "conversion")

        # 3. Traffic Source Performance
        if c["source"] and (c["orders"] or c["sessions"]):
            metric = c["orders"] if c["orders"] else c["sessions"]
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["source"])[metric].sum().nlargest(7).sort_values().plot(kind="barh", ax=ax, color="#9467bd")
            ax.set_title(f"Top Sources by {metric.replace('_',' ').title()}")
            save(fig, "sources.png", "Channel performance", 0.8, "acquisition")

        # 4. Device Mix (FIXED: Added axis('equal'))
        if c["device"] and c["sessions"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["device"])[c["sessions"]].sum().head(5).plot(kind="pie", ax=ax, autopct='%1.1f%%')
            ax.axis('equal') # Ensures pie is round
            ax.set_ylabel("")
            ax.set_title("Traffic by Device")
            save(fig, "device_mix.png", "User platform", 0.7, "tech")

        # 5. AOV Distribution (FIXED: Safe filtering)
        if c["revenue"] and c["orders"]:
            # Filter for actual orders to avoid skew
            mask = df[c["orders"]] > 0
            if mask.sum() > 10:
                fig, ax = plt.subplots(figsize=(6, 4))
                # Calculate per-order value safely
                (df.loc[mask, c["revenue"]] / df.loc[mask, c["orders"]]).hist(ax=ax, bins=20, color="green")
                ax.set_title("Order Value Distribution")
                save(fig, "aov_dist.png", "Spending habits", 0.75, "economics")

        # 6. Bounce Rate Trend
        if self.time_col and c["bounce"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(self.time_col).resample('M')[c["bounce"]].mean().plot(ax=ax, color="red")
            ax.set_title("Bounce Rate Trend")
            ax.set_ylim(0, 1)
            imp = 0.9 if kpis.get("avg_bounce_rate", 0) > 0.6 else 0.65
            save(fig, "bounce_trend.png", "Traffic quality", imp, "engagement")

        # 7. Return Rate by Category
        if c["returns"] and c["category"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["category"])[c["returns"]].sum().nlargest(5).plot(kind="bar", ax=ax, color="maroon")
            ax.set_title("Returns by Category")
            save(fig, "returns_cat.png", "Product quality issues", 0.88, "product")

        # 8. Conversion vs AOV (FIXED: Zero guards)
        if c["source"] and c["revenue"] and c["sessions"]:
            # Aggregate first
            agg = df.groupby(c["source"]).agg({c["revenue"]: "sum", c["orders"]: "sum", c["sessions"]: "sum"})
            
            # Filter valid orders
            agg = agg[agg[c["orders"]] > 0]
            
            if not agg.empty:
                agg["cr"] = agg[c["orders"]] / agg[c["sessions"]]
                agg["aov"] = agg[c["revenue"]] / agg[c["orders"]]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(agg["cr"], agg["aov"], alpha=0.6)
                ax.set_xlabel("Conversion Rate")
                ax.set_ylabel("AOV")
                ax.set_title("Source Quality: CR vs AOV")
                save(fig, "cr_vs_aov.png", "Channel quality matrix", 0.82, "marketing")

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        # 1. Composite Insights (The Smart Layer)
        composite = self.generate_composite_insights(df, kpis)
        insights += composite
        
        titles = [i["title"] for i in composite]
        
        # 2. Atomic Fallbacks
        cr = kpis.get("conversion_rate")
        bounce = kpis.get("avg_bounce_rate")
        abandonment = kpis.get("cart_abandonment_rate")
        
        # CR Alert
        if cr is not None and cr < 0.015:
            if not any("Empty Calorie" in t for t in titles):
                insights.append({
                    "level": "RISK", "title": "Low Conversion",
                    "so_what": f"Conversion Rate is {cr:.2%}, below 1.5% benchmark."
                })

        # Abandonment Alert
        if abandonment is not None and abandonment > 0.75:
            if not any("Payment Gateway" in t for t in titles):
                insights.append({
                    "level": "WARNING", "title": "High Cart Abandonment",
                    "so_what": f"{abandonment:.1%} of carts are abandoned."
                })

        # Bounce Alert
        if bounce is not None and bounce > 0.70:
            insights.append({
                "level": "WARNING", "title": "High Bounce Rate",
                "so_what": f"Bounce rate is {bounce:.1%}. Landing pages may be irrelevant."
            })

        if not insights:
            insights.append({"level": "INFO", "title": "Performance Stable", "so_what": "Metrics are healthy."})

        return insights

    # ---------------- COMPOSITE INSIGHTS ----------------

    def generate_composite_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        sessions = kpis.get("total_sessions", 0)
        cr = kpis.get("conversion_rate")
        abandonment = kpis.get("cart_abandonment_rate")
        checkout_drop = kpis.get("checkout_dropoff_rate")

        # 1. Empty Calorie Traffic (Bot/Fraud Logic Tweak)
        if sessions > 500 and cr is not None and cr < 0.005:
            insights.append({
                "level": "CRITICAL", "title": "Empty Calorie Traffic (Possible Bots)",
                "so_what": f"High traffic ({sessions:,.0f}) but near-zero conversion ({cr:.2%})."
            })

        # 2. Payment Friction (Low Abandonment + High Checkout Drop)
        if abandonment is not None and checkout_drop is not None:
            if abandonment < 0.60 and checkout_drop > 0.70:
                insights.append({
                    "level": "CRITICAL", "title": "Payment Gateway Friction",
                    "so_what": "Users add to cart, but 70%+ drop off at payment step."
                })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if "Payment Gateway Friction" in titles:
            recs.append({"action": "Audit payment gateway for technical errors immediately.", "priority": "HIGH"})

        if "Empty Calorie Traffic" in titles:
            recs.append({"action": "Review ad targeting and block bot IPs.", "priority": "HIGH"})

        if kpis.get("cart_abandonment_rate", 0) > 0.75:
            recs.append({"action": "Enable abandoned cart recovery emails.", "priority": "MEDIUM"})

        return recs or [{"action": "Optimize landing pages.", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class EcommerceDomainDetector(BaseDomainDetector):
    domain_name = "ecommerce"
    TOKENS = {"session", "pageview", "bounce", "cart", "checkout", "conversion", "traffic", "referrer"}

    def detect(self, df) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # Boost if Session + Cart exist (Classic E-com signature)
        cols = str(df.columns).lower()
        if "session" in cols and "cart" in cols:
            confidence = max(confidence, 0.95)
            
        return DomainDetectionResult("ecommerce", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("ecommerce", EcommerceDomain, EcommerceDomainDetector)
