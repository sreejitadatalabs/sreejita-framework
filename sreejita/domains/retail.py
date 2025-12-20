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
    """Safely divides n by d, returning None if d is 0 or NaN."""
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detects a time column suitable for retail analytics.
    """
    candidates = [
        "order date", "invoice date", "transaction date",
        "date", "order_date", "ship date"
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
# RETAIL DOMAIN (v3.1 - ENTERPRISE INTELLIGENCE)
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Retail & Sales Analytics (Revenue, Profit, Shipping, Discount)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        return (
            resolve_column(df, "revenue") is not None
            or resolve_column(df, "sales") is not None
            or resolve_column(df, "amount") is not None
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
        kpis = {}

        # 1. Resolve Columns
        revenue_col = (
            resolve_column(df, "revenue")
            or resolve_column(df, "sales")
            or resolve_column(df, "amount")
        )
        order_id = resolve_column(df, "order_id") or resolve_column(df, "transaction")
        customer = resolve_column(df, "customer")
        category = resolve_column(df, "category") or resolve_column(df, "product")
        
        # Profit / Margin
        profit_col = resolve_column(df, "profit") 
        margin_col = resolve_column(df, "margin")
        
        # Operational
        shipping_col = resolve_column(df, "shipping_cost") or resolve_column(df, "freight")
        discount_col = resolve_column(df, "discount")
        stock_col = resolve_column(df, "stock") or resolve_column(df, "inventory")

        # 2. Base Metrics
        if revenue_col:
            total_rev = df[revenue_col].sum()
            kpis["total_revenue"] = total_rev
            
            # AOV
            if order_id and df[order_id].notna().any():
                kpis["total_orders"] = df[order_id].nunique()
                kpis["avg_order_value"] = _safe_div(total_rev, kpis["total_orders"])

            # ENTERPRISE FIX: Robust Revenue Growth
            if self.has_time_series:
                try:
                    # Create temp series
                    ts = df.set_index(self.time_col)[revenue_col]
                    span_days = (df[self.time_col].max() - df[self.time_col].min()).days
                    
                    # Aggregation: Monthly for long history, Weekly for short
                    # This smooths out "Sunday vs Monday" noise
                    if span_days > 90:
                        resampled = ts.resample('ME').sum()
                    else:
                        resampled = ts.resample('W').sum()
                    
                    # Filter empty periods
                    resampled = resampled[resampled > 0]
                    
                    if len(resampled) >= 2:
                        first_val = resampled.iloc[0]
                        last_val = resampled.iloc[-1]
                        kpis["revenue_growth"] = (last_val - first_val) / first_val
                    else:
                        # Fallback to simple point-to-point if not enough periods
                        first_val = df.iloc[0][revenue_col]
                        last_val = df.iloc[-1][revenue_col]
                        if first_val != 0:
                            kpis["revenue_growth"] = (last_val - first_val) / first_val
                except Exception:
                    pass

            # Concentration
            if category:
                top_cat_rev = df.groupby(category)[revenue_col].sum().max()
                kpis["top_category_revenue_share"] = _safe_div(top_cat_rev, total_rev)

            # Profitability
            kpis["target_profit_margin"] = 0.15 

            if profit_col:
                total_profit = df[profit_col].sum()
                kpis["total_profit"] = total_profit
                kpis["profit_margin"] = _safe_div(total_profit, total_rev)
            
            elif margin_col:
                avg_margin = df[margin_col].mean()
                if avg_margin > 1: avg_margin = avg_margin / 100
                kpis["profit_margin"] = avg_margin

            # Shipping
            if shipping_col:
                total_shipping = df[shipping_col].sum()
                kpis["shipping_cost_ratio"] = _safe_div(total_shipping, total_rev)

        # 3. Stockout Rate
        if stock_col and pd.api.types.is_numeric_dtype(df[stock_col]):
            kpis["stockout_rate"] = (df[stock_col] <= 0).mean()

        if customer:
            kpis["customer_count"] = df[customer].nunique()

        # 4. Discount Metrics
        if discount_col:
            avg_disc = df[discount_col].mean()
            if avg_disc > 1: avg_disc = avg_disc / 100
            kpis["average_discount"] = avg_disc
            kpis["avg_discount_pct"] = avg_disc 

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        kpis = self.calculate_kpis(df)

        def human_fmt(x, _):
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        revenue = (
            resolve_column(df, "revenue")
            or resolve_column(df, "sales")
            or resolve_column(df, "amount")
        )

        # Safety Guard
        if revenue and not pd.api.types.is_numeric_dtype(df[revenue]):
            return visuals

        category = resolve_column(df, "category") or resolve_column(df, "product")
        customer = resolve_column(df, "customer")
        discount = resolve_column(df, "discount")

        # --- Visual 1: Sales Trend ---
        if self.has_time_series and revenue:
            p = output_dir / "sales_trend.png"
            plt.figure(figsize=(7, 4))
            
            plot_df = df.copy()
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample('ME')
                    .sum()
                    .reset_index()
                )

            plt.plot(plot_df[self.time_col], plot_df[revenue], linewidth=2, color="#1f77b4")
            plt.title("Sales Trend")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Sales performance over time"})

        # --- Visual 2: Revenue by Category ---
        if revenue and category:
            p = output_dir / "revenue_by_category.png"
            top_cats = df.groupby(category)[revenue].sum().sort_values(ascending=False).head(7)
            
            plt.figure(figsize=(7, 4))
            top_cats.plot(kind="bar", color="#ff7f0e")
            plt.title("Revenue by Category")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Top revenue-generating categories"})

        # --- Visual 3: Top Customers ---
        if revenue and customer:
            p = output_dir / "top_customers.png"
            top_cust = df.groupby(customer)[revenue].sum().sort_values(ascending=True).tail(7)
            
            plt.figure(figsize=(7, 4))
            top_cust.plot(kind="barh", color="#2ca02c")
            plt.title("Top Customers by Revenue")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Customer revenue concentration"})

        # --- Visual 4: Discount Impact ---
        if (
            discount 
            and category 
            and pd.api.types.is_numeric_dtype(df[discount])
            and df[discount].notna().any()
        ):
            p = output_dir / "discount_by_category.png"
            disc_by_cat = df.groupby(category)[discount].mean().sort_values(ascending=False).head(7)
            
            if disc_by_cat.max() > 1:
                disc_by_cat = disc_by_cat / 100

            if disc_by_cat.max() > 0.01:
                plt.figure(figsize=(7, 4))
                disc_by_cat.plot(kind="bar", color="#d62728")
                plt.title("Avg Discount Level by Category")
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(p)
                plt.close()
                visuals.append({"path": p, "caption": "Categories with highest discount rates"})

        return visuals[:4]

    # ---------------- ATOMIC INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        margin = kpis.get("profit_margin")
        ship_ratio = kpis.get("shipping_cost_ratio")
        avg_disc = kpis.get("average_discount")

        # 1. Profitability Insights
        if margin is not None:
            if margin < 0:
                insights.append({
                    "level": "RISK",
                    "title": "Negative Profit Margin",
                    "so_what": f"You are losing money on sales. Margin is {margin:.1%}."
                })
            elif margin < 0.10: 
                insights.append({
                    "level": "WARNING",
                    "title": "Low Profit Margin",
                    "so_what": f"Margin is {margin:.1%}, below the 15% target."
                })

        # 2. Discounting Insights
        if avg_disc is not None:
            if avg_disc > 0.25:
                insights.append({
                    "level": "WARNING",
                    "title": "Heavy Discounting Detected",
                    "so_what": f"Avg discount is {avg_disc:.1%}, potentially eroding brand value."
                })

        # 3. Shipping Cost Insights
        if ship_ratio is not None:
            if ship_ratio > 0.15:
                insights.append({
                    "level": "WARNING",
                    "title": "High Shipping Costs",
                    "so_what": f"Shipping consumes {ship_ratio:.1%} of revenue (Target: <10%)."
                })

        # === CALL COMPOSITE LAYER (v3.1) ===
        insights += self.generate_composite_insights(df, kpis)

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Retail Performance Stable",
                "so_what": "Key metrics (Margin, Shipping, Discounts) are within healthy ranges."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (RETAIL v3.1) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retail v3.1 Composite Intelligence Layer.
        """
        insights: List[Dict[str, Any]] = []

        revenue_growth = kpis.get("revenue_growth")
        margin = kpis.get("profit_margin")
        discount = kpis.get("avg_discount_pct")
        orders = kpis.get("total_orders")
        aov = kpis.get("avg_order_value")
        stockout_rate = kpis.get("stockout_rate")
        top_category_share = kpis.get("top_category_revenue_share")

        # 1. Revenue Growth + Margin Decline
        if revenue_growth is not None and margin is not None:
            if revenue_growth > 0.10 and margin < 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Revenue Growth Driven by Margin Compression",
                    "so_what": (
                        f"Revenue is growing ({revenue_growth:.1%}), but profit margin is low "
                        f"({margin:.1%}). Growth appears discount-driven rather than demand-led."
                    )
                })

        # 2. High Discount + Flat Orders
        if discount is not None and orders is not None:
            if discount > 0.20 and revenue_growth is not None and revenue_growth < 0.05:
                insights.append({
                    "level": "WARNING",
                    "title": "Discounting Not Translating to Demand",
                    "so_what": (
                        f"Average discount is high ({discount:.1%}), but order growth remains weak. "
                        f"Promotions may be inefficient or poorly targeted."
                    )
                })

        # 3. High Orders + Low AOV (Dynamic Threshold)
        if orders is not None and aov is not None:
            # v3.1 Logic: If orders are substantial (>50% of dataset rows) but AOV is low
            if orders > df.shape[0] * 0.5 and aov < 20: 
                insights.append({
                    "level": "INFO",
                    "title": "High Volume, Small Basket Size",
                    "so_what": (
                        f"Order volume is strong, but average order value is low "
                        f"({aov:.2f}). Fulfillment costs may be disproportionately high."
                    )
                })

        # 4. Stockouts During High Demand
        if stockout_rate is not None and revenue_growth is not None:
            if stockout_rate > 0.10 and revenue_growth > 0.10:
                insights.append({
                    "level": "RISK",
                    "title": "Demand Outpacing Inventory Availability",
                    "so_what": (
                        f"Stockout rate is elevated ({stockout_rate:.1%}) while demand is rising. "
                        f"Inventory gaps may be causing lost sales."
                    )
                })

        # 5. Category Concentration Risk
        if top_category_share is not None:
            if top_category_share > 0.60:
                insights.append({
                    "level": "WARNING",
                    "title": "Revenue Concentration Risk",
                    "so_what": (
                        f"Over {top_category_share:.0%} of revenue comes from a single category. "
                        f"This increases exposure to demand or supply disruptions."
                    )
                })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        margin = kpis.get("profit_margin")
        avg_disc = kpis.get("average_discount")
        ship_ratio = kpis.get("shipping_cost_ratio")

        if margin is not None and margin < 0.05:
            recs.append({
                "action": "Audit product pricing and cost of goods sold (COGS)",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
        
        if avg_disc is not None and avg_disc > 0.25:
            recs.append({
                "action": "Tighten discount approval thresholds",
                "priority": "HIGH",
                "timeline": "Next Month"
            })

        if ship_ratio is not None and ship_ratio > 0.12:
            recs.append({
                "action": "Renegotiate carrier rates or adjust free shipping threshold",
                "priority": "MEDIUM",
                "timeline": "This Quarter"
            })

        if not recs:
            recs.append({
                "action": "Continue monitoring sales performance",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"

    RETAIL_TOKENS: Set[str] = {
        "order", "sales", "revenue", "customer",
        "product", "sku", "category", "discount",
        "shipping", "margin", "inventory", "stock"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.RETAIL_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

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

