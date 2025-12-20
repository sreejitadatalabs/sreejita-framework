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
    Retail-safe time detector.
    """
    candidates = [
        "order_date", "date", "transaction_date", 
        "day", "month", "year", "invoice_date"
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
# RETAIL DOMAIN (v3.0 - FULL AUTHORITY)
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Retail & Sales Analytics (Revenue, Margin, Discounts, AOV)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Retail data requires Sales/Revenue info OR Order/Product specifics.
        """
        return any(
            resolve_column(df, c) is not None
            for c in [
                "sales", "revenue", "profit", "margin",
                "discount", "quantity", "order_id", "product_id",
                "category", "segment", "shipping_cost"
            ]
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

        # Column Resolution
        sales = resolve_column(df, "sales") or resolve_column(df, "revenue")
        profit = resolve_column(df, "profit") or resolve_column(df, "margin")
        discount = resolve_column(df, "discount")
        shipping = resolve_column(df, "shipping_cost")
        order_id = resolve_column(df, "order_id")
        
        # 1. Revenue & Growth
        if sales and pd.api.types.is_numeric_dtype(df[sales]):
            kpis["total_revenue"] = df[sales].sum()
            
            # Simple Growth Calculation if Time Series exists
            if self.has_time_series:
                # Compare first 20% vs last 20% of time period to estimate trend
                n = len(df)
                start_rev = df[sales].iloc[:n//5].mean()
                end_rev = df[sales].iloc[-n//5:].mean()
                if start_rev > 0:
                    kpis["revenue_growth_est"] = (end_rev - start_rev) / start_rev

        # 2. Profit Margin
        if profit and pd.api.types.is_numeric_dtype(df[profit]):
            total_profit = df[profit].sum()
            kpis["total_profit"] = total_profit
            
            if "total_revenue" in kpis:
                kpis["profit_margin"] = _safe_div(total_profit, kpis["total_revenue"])

        # 3. Discounting
        if discount and pd.api.types.is_numeric_dtype(df[discount]):
            kpis["average_discount"] = df[discount].mean()

        # 4. Shipping Cost Ratio
        if shipping and "total_revenue" in kpis and pd.api.types.is_numeric_dtype(df[shipping]):
            total_shipping = df[shipping].sum()
            kpis["shipping_cost_ratio"] = _safe_div(total_shipping, kpis["total_revenue"])

        # 5. AOV (Average Order Value)
        if order_id and "total_revenue" in kpis:
            order_count = df[order_id].nunique()
            kpis["aov"] = _safe_div(kpis["total_revenue"], order_count)

        return kpis

    # ---------------- VISUALS (MAX 4) ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        kpis = self.calculate_kpis(df)

        def human_fmt(x, _):
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        sales = resolve_column(df, "sales") or resolve_column(df, "revenue")
        profit = resolve_column(df, "profit")
        category = resolve_column(df, "category") or resolve_column(df, "sub_category")
        segment = resolve_column(df, "segment")
        region = resolve_column(df, "region")

        # -------- Visual 1: Sales Trend --------
        if self.has_time_series and sales and pd.api.types.is_numeric_dtype(df[sales]):
            p = output_dir / "sales_trend.png"
            plt.figure(figsize=(7, 4))
            
            plot_df = df.copy()
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME")
                    .sum()
                    .reset_index()
                )
            
            plt.plot(plot_df[self.time_col], plot_df[sales], linewidth=2, color="#2ca02c")
            plt.title("Revenue Trend Over Time")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Sales performance timeline"})

        # -------- Visual 2: Profit by Category --------
        if category and profit and pd.api.types.is_numeric_dtype(df[profit]):
            p = output_dir / "profit_by_category.png"
            
            top_cats = df.groupby(category)[profit].sum().sort_values(ascending=False).head(7)
            
            plt.figure(figsize=(7, 4))
            top_cats.plot(kind="bar", color="#1f77b4")
            plt.title("Profit by Category")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Most profitable product categories"})

        # -------- Visual 3: Sales by Segment (Pie) --------
        if segment and sales and pd.api.types.is_numeric_dtype(df[sales]):
            p = output_dir / "sales_by_segment.png"
            
            seg_sales = df.groupby(segment)[sales].sum().sort_values(ascending=False).head(5)
            
            plt.figure(figsize=(6, 4))
            seg_sales.plot(kind="pie", autopct='%1.1f%%')
            plt.ylabel("")
            plt.title("Sales by Customer Segment")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Revenue share by customer segment"})

        return visuals[:4]

    # ---------------- ATOMIC INSIGHTS (WITH DOMINANCE RULE) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        # === STEP 1: Composite FIRST (Authority Layer) ===
        composite: List[Dict[str, Any]] = []
        if len(df) > 30:
            composite = self.generate_composite_insights(df, kpis)

        dominant_titles = {
            i["title"] for i in composite
            if i["level"] in {"RISK", "WARNING"}
        }

        # === STEP 2: Suppression Rules ===
        suppress_margin = "Revenue Growth Driven by Margin Compression" in dominant_titles
        suppress_discount = "Discounting Not Translating to Demand" in dominant_titles

        margin = kpis.get("profit_margin")
        avg_disc = kpis.get("average_discount")
        ship_ratio = kpis.get("shipping_cost_ratio")

        # === STEP 3: Guarded Atomic Insights ===
        
        # Profit Margin
        if margin is not None and not suppress_margin:
            if margin < 0:
                insights.append({
                    "level": "RISK",
                    "title": "Negative Profitability",
                    "so_what": f"Overall profit margin is {margin:.1%}. The business is operating at a loss."
                })
            elif margin < 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Low Profit Margins",
                    "so_what": f"Profit margin is {margin:.1%}, below healthy retail benchmarks (10%+)."
                })

        # Discounting
        if avg_disc is not None and not suppress_discount:
            if avg_disc > 0.25:
                insights.append({
                    "level": "WARNING",
                    "title": "Aggressive Discounting",
                    "so_what": f"Average discount is {avg_disc:.1%}, which may erode brand value."
                })

        # Shipping Costs
        if ship_ratio is not None:
            if ship_ratio > 0.15:
                insights.append({
                    "level": "WARNING",
                    "title": "High Shipping Costs",
                    "so_what": f"Shipping consumes {ship_ratio:.1%} of total revenue."
                })

        # === STEP 4: Composite LAST (Authority Wins) ===
        insights += composite

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Retail Performance Stable",
                "so_what": "Sales, margin, and discount metrics are within expected ranges."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (RETAIL v3.0) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retail v3 Composite Intelligence Layer.
        """
        insights: List[Dict[str, Any]] = []

        growth = kpis.get("revenue_growth_est")
        margin = kpis.get("profit_margin")
        discount = kpis.get("average_discount")
        
        # 1. Revenue Growth Driven by Margin Compression
        # (Sales are up, but we are bleeding profit to get them)
        if growth is not None and margin is not None:
            if growth > 0.10 and margin < 0.05:
                insights.append({
                    "level": "RISK",
                    "title": "Revenue Growth Driven by Margin Compression",
                    "so_what": (
                        f"Revenue is growing ({growth:.1%}) but margins have collapsed "
                        f"({margin:.1%}). You are buying growth with profitability."
                    )
                })

        # 2. Discounting Not Translating to Demand
        # (High discounts, but low/negative growth)
        if discount is not None and growth is not None:
            if discount > 0.20 and growth < 0.02:
                insights.append({
                    "level": "WARNING",
                    "title": "Discounting Not Translating to Demand",
                    "so_what": (
                        f"Aggressive discounting ({discount:.1%}) is failing to drive "
                        f"significant revenue growth ({growth:.1%}). Strategy ineffective."
                    )
                })
        
        # 3. Demand Outpacing Inventory (Proxy)
        # If we have growth but zero inventory checks (Assuming inventory cols might exist in some datasets)
        # This is a placeholder for datasets that might have 'stock_level'
        inventory = resolve_column(df, "quantity_in_stock")
        if growth is not None and inventory:
             # Just an example logic: High growth + Low Stock avg
             if growth > 0.20:
                 insights.append({
                    "level": "INFO",
                    "title": "High Demand Velocity",
                    "so_what": "Sales are growing rapidly. Ensure inventory replenishment keeps pace."
                })

        return insights

    # ---------------- RECOMMENDATIONS (AUTHORITY BASED) ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        # 1. Check Composite Context
        composite = []
        if len(df) > 30:
            composite = self.generate_composite_insights(df, kpis)
        
        titles = [i["title"] for i in composite]

        # AUTHORITY RULES: Specific composites mandate specific actions
        if "Revenue Growth Driven by Margin Compression" in titles:
            return [{
                "action": "Reduce blanket discounting and reprice high-volume SKUs to restore margin",
                "priority": "HIGH",
                "timeline": "Immediate"
            }]

        if "Discounting Not Translating to Demand" in titles:
            return [{
                "action": "Stop ineffective promotions and redesign targeted offers",
                "priority": "HIGH",
                "timeline": "Next Campaign"
            }]

        # 2. Fallback to Atomic Recs
        margin = kpis.get("profit_margin")
        discount = kpis.get("average_discount")

        if margin is not None and margin < 0.10:
            recs.append({
                "action": "Conduct SKU profitability audit to identify loss-leaders",
                "priority": "HIGH",
                "timeline": "This Week"
            })
        
        if discount is not None and discount > 0.25:
             recs.append({
                "action": "Review pricing strategy to reduce reliance on heavy discounting",
                "priority": "MEDIUM",
                "timeline": "Next Quarter"
            })

        if not recs:
            recs.append({
                "action": "Continue optimizing product mix and sales channels",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR (COLLISION PROOF)
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"

    RETAIL_TOKENS: Set[str] = {
        "sales", "revenue", "profit", "margin",
        "discount", "quantity", "order_id", "product",
        "category", "segment", "shipping", "customer", "store"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        
        hits = [c for c in cols if any(t in c for t in self.RETAIL_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        # 🔑 RETAIL DOMINANCE RULE (Collision Proofing)
        # Distinguish from Supply Chain (Inventory focus) & Marketing (Campaign focus)
        retail_exclusive = any(
            t in c for c in cols
            for t in {"order_id", "customer", "avg_order", "discount", "margin", "store"}
        )

        if retail_exclusive:
            confidence = max(confidence, 0.85)

        return DomainDetectionResult(
            domain="retail",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="retail",
        domain_cls=RetailDomain,
        detector_cls=RetailDomainDetector,
    )
