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
    Prioritizes specific retail terms like 'order date'.
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
                    # Validate by attempting to parse a small sample
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
    description = "Retail & Sales Analytics (Revenue, Orders, Products, Customers)"

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
        Calculates key retail metrics: Total Revenue, AOV, Order Volume.
        """
        kpis = {}

        revenue = (
            resolve_column(df, "revenue")
            or resolve_column(df, "sales")
            or resolve_column(df, "amount")
        )

        order_id = resolve_column(df, "order_id") or resolve_column(df, "transaction")
        customer = resolve_column(df, "customer")

        if revenue:
            kpis["total_revenue"] = df[revenue].sum()
            
            # --- YOUR FIX: Validate Order ID is usable ---
            if order_id and df[order_id].notna().any():
                kpis["avg_order_value"] = _safe_div(
                    df[revenue].sum(), df[order_id].nunique()
                )

        if order_id:
            kpis["order_volume"] = df[order_id].nunique()

        if customer:
            kpis["customer_count"] = df[customer].nunique()

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Generates visualizations for the report.
        Includes safeguards for data types and aggregation for large datasets.
        """

        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)

        def human_fmt(x, _):
            """Format numbers for human readability (e.g., 1.5M, 10K)."""
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        revenue = (
            resolve_column(df, "revenue")
            or resolve_column(df, "sales")
            or resolve_column(df, "amount")
        )

        # --- SAFETY GUARD: Numeric Check ---
        # If revenue column exists but contains non-numeric data (e.g., "$1,200"),
        # stop generation to prevent crashes.
        if revenue and not pd.api.types.is_numeric_dtype(df[revenue]):
            return visuals
        # -----------------------------------

        category = resolve_column(df, "category") or resolve_column(df, "product")
        customer = resolve_column(df, "customer")

        # --- Visual 1: Sales Trend (Smart Aggregation) ---
        if self.has_time_series and revenue:
            p = output_dir / "sales_trend.png"
            plt.figure(figsize=(7, 4))
            
            # --- AGGREGATION LOGIC ---
            plot_df = df.copy()
            # If dataset is large (>100 rows), aggregate to Month End ('ME')
            # to prevent messy charts.
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample('ME')
                    .sum()
                    .reset_index()
                )
            # -------------------------

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
            
            # Get Top 7 Categories
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
            
            # Get Top 7 Customers
            top_cust = df.groupby(customer)[revenue].sum().sort_values(ascending=True).tail(7)
            
            plt.figure(figsize=(7, 4))
            top_cust.plot(kind="barh", color="#2ca02c")
            plt.title("Top Customers by Revenue")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Customer revenue concentration"})

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates text insights based on calculated KPIs.
        """
        insights = []

        revenue = kpis.get("total_revenue")
        orders = kpis.get("order_volume")
        aov = kpis.get("avg_order_value")

        if revenue and orders and aov:
            insights.append({
                "level": "INFO",
                "title": "Revenue Composition",
                "so_what": (
                    f"Revenue is driven by {orders} orders "
                    f"with an average order value of {aov:,.2f}."
                )
            })

        # Default insight if nothing specific is found
        return insights or [{
            "level": "INFO",
            "title": "Retail Performance Stable",
            "so_what": "Sales metrics fall within expected operating ranges."
        }]

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates actionable recommendations based on KPIs.
        """
        recs = []

        # Simple threshold-based recommendation for AOV
        if kpis.get("avg_order_value") and kpis["avg_order_value"] < 50:
            recs.append({
                "action": "Increase average order value through bundling or upsell",
                "priority": "MEDIUM",
                "timeline": "Next Quarter"
            })

        return recs or [{
            "action": "Continue monitoring sales performance",
            "priority": "LOW",
            "timeline": "Ongoing"
        }]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"

    RETAIL_TOKENS: Set[str] = {
        "order", "sales", "revenue", "customer",
        "product", "sku", "category", "discount"
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
