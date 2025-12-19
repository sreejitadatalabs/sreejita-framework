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


# =====================================================
# RETAIL DOMAIN
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Retail & Sales Analytics (Revenue, Profit, Shipping, Discount)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validates if the dataset contains the minimum required columns.
        """
        return (
            resolve_column(df, "revenue") is not None
            or resolve_column(df, "sales") is not None
            or resolve_column(df, "amount") is not None
        )

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the dataframe by detecting and parsing the time column.
        """
        self.time_col = _detect_time_column(df)
        self.has_time_series = False

        if self.time_col:
            df = df.copy() # Avoid SettingWithCopy warnings
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.dropna(subset=[self.time_col])
            df = df.sort_values(self.time_col)
            # Ensure we have enough data points for a time series
            self.has_time_series = df[self.time_col].nunique() >= 2

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates key retail metrics including Profit, Shipping, and Discounts.
        """
        kpis = {}

        # 1. Resolve Columns
        revenue_col = (
            resolve_column(df, "revenue")
            or resolve_column(df, "sales")
            or resolve_column(df, "amount")
        )
        order_id = resolve_column(df, "order_id") or resolve_column(df, "transaction")
        customer = resolve_column(df, "customer")
        
        profit_col = resolve_column(df, "profit") or resolve_column(df, "margin")
        shipping_col = resolve_column(df, "shipping_cost") or resolve_column(df, "freight")
        discount_col = resolve_column(df, "discount")

        # 2. Base Metrics
        if revenue_col:
            total_rev = df[revenue_col].sum()
            kpis["total_revenue"] = total_rev
            
            if order_id and df[order_id].notna().any():
                kpis["avg_order_value"] = _safe_div(total_rev, df[order_id].nunique())

            # 3. Profitability Metrics
            if profit_col:
                total_profit = df[profit_col].sum()
                kpis["total_profit"] = total_profit
                kpis["profit_margin"] = _safe_div(total_profit, total_rev)
                kpis["target_profit_margin"] = 0.15  # Standard Retail Target (15%)

            # 4. Cost Metrics
            if shipping_col:
                total_shipping = df[shipping_col].sum()
                kpis["shipping_cost_ratio"] = _safe_div(total_shipping, total_rev)
                kpis["target_shipping_ratio"] = 0.10 # Target < 10%

        if order_id:
            kpis["order_volume"] = df[order_id].nunique()

        if customer:
            kpis["customer_count"] = df[customer].nunique()

        # 5. Discount Metrics
        if discount_col:
            # Assuming discount is a ratio (0.1) or percentage. Mean gives avg rate.
            kpis["average_discount"] = df[discount_col].mean()

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Generates visualizations: Trend, Category, Customers, and Discount Analysis.
        """

        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        def human_fmt(x, _):
            """Format numbers for human readability."""
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        revenue = (
            resolve_column(df, "revenue")
            or resolve_column(df, "sales")
            or resolve_column(df, "amount")
        )

        # --- SAFETY GUARD ---
        if revenue and not pd.api.types.is_numeric_dtype(df[revenue]):
            return visuals
        # --------------------

        category = resolve_column(df, "category") or resolve_column(df, "product")
        customer = resolve_column(df, "customer")
        discount = resolve_column(df, "discount")

        # --- Visual 1: Sales Trend (Smart Aggregation) ---
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

        # --- Visual 4: Discount Impact (Profit Leak Analysis) ---
        if discount and category and pd.api.types.is_numeric_dtype(df[discount]):
            p = output_dir / "discount_by_category.png"
            
            # Avg Discount by Category
            disc_by_cat = df.groupby(category)[discount].mean().sort_values(ascending=False).head(7)
            
            # Only plot if meaningful discounts exist (> 1%)
            if disc_by_cat.max() > 0.01:
                plt.figure(figsize=(7, 4))
                disc_by_cat.plot(kind="bar", color="#d62728") # Red for "Cost/Loss"
                plt.title("Avg Discount Level by Category")
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(p)
                plt.close()
                visuals.append({"path": p, "caption": "Categories with highest discount rates"})

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates insights based on Retail Thresholds.
        """
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
            elif margin < 0.10: # Below 10% is thin for retail
                insights.append({
                    "level": "WARNING",
                    "title": "Low Profit Margin",
                    "so_what": f"Margin is {margin:.1%}, below the 15% target."
                })

        # 2. Shipping Cost Insights
        if ship_ratio is not None:
            if ship_ratio > 0.15: # >15% is very high
                insights.append({
                    "level": "WARNING",
                    "title": "High Shipping Costs",
                    "so_what": f"Shipping consumes {ship_ratio:.1%} of revenue (Target: <10%)."
                })

        # 3. Discounting Insights
        if avg_disc is not None:
            if avg_disc > 0.25: # >25% average discount is aggressive
                insights.append({
                    "level": "WARNING",
                    "title": "Heavy Discounting Detected",
                    "so_what": f"Avg discount is {avg_disc:.1%}, eroding margins."
                })

        # Default Info
        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Retail Performance Stable",
                "so_what": "Key metrics (Margin, Shipping, Discounts) are within healthy ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates actionable recommendations based on KPIs.
        """
        recs = []

        margin = kpis.get("profit_margin")
        ship_ratio = kpis.get("shipping_cost_ratio")
        avg_disc = kpis.get("average_discount")
        aov = kpis.get("avg_order_value")

        # Prioritized Actions
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

        if aov and aov < 50:
            recs.append({
                "action": "Implement bundles to boost Average Order Value",
                "priority": "LOW",
                "timeline": "Next Quarter"
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
        "shipping", "margin"
    }

    def detect(self, df) -> DomainDetectionResult:
        """
        Detects if the dataframe belongs to the Retail domain.
        """
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
