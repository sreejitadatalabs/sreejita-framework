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
    Supply-chain-safe time detector.
    Priority: order -> ship -> delivery.
    """
    candidates = [
        "order date", "order_date",
        "ship date", "shipment date",
        "delivery date", "delivered date",
        "date"
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
# SUPPLY CHAIN / OPERATIONS DOMAIN
# =====================================================

class SupplyChainDomain(BaseDomain):
    name = "supply_chain"
    description = "Operational & Supply Chain Analytics (Delivery, Inventory, Fulfillment)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        # Flexible validation: Needs either Order info OR Inventory info
        has_orders = resolve_column(df, "order_id") or resolve_column(df, "shipment")
        has_stock = resolve_column(df, "inventory") or resolve_column(df, "stock")
        
        return bool(has_orders or has_stock)

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

        order_id = resolve_column(df, "order_id") or resolve_column(df, "shipment")
        inventory = resolve_column(df, "inventory") or resolve_column(df, "stock")
        sku = resolve_column(df, "sku") or resolve_column(df, "product")

        order_date = resolve_column(df, "order_date")
        delivery_date = resolve_column(df, "delivery_date") or resolve_column(df, "delivered")
        promised_date = resolve_column(df, "promised_date") or resolve_column(df, "expected_date")
        status_col = resolve_column(df, "status")

        # 1. Order Volume
        if order_id:
            kpis["order_volume"] = df[order_id].nunique()

        # 2. Lead Time (Actual Calculation)
        if order_date and delivery_date:
            try:
                # Ensure datetime types
                start = pd.to_datetime(df[order_date], errors='coerce')
                end = pd.to_datetime(df[delivery_date], errors='coerce')
                lead_time = (end - start).dt.days
                kpis["avg_lead_time_days"] = lead_time.mean()
            except Exception:
                pass

        # 3. On-Time Delivery (The Hybrid Approach)
        kpis["target_otd_rate"] = 0.95 # Benchmark

        if delivery_date and promised_date:
            # Gold Standard: Date Math
            try:
                delivered = pd.to_datetime(df[delivery_date], errors='coerce')
                promised = pd.to_datetime(df[promised_date], errors='coerce')
                # Check valid dates only
                mask = delivered.notna() & promised.notna()
                if mask.sum() > 0:
                    on_time = (delivered[mask] <= promised[mask]).mean()
                    kpis["on_time_delivery_rate"] = on_time
            except Exception:
                pass
        
        # Fallback: Status Text Check (If date math failed or columns missing)
        if "on_time_delivery_rate" not in kpis and status_col:
            # Look for negative keywords
            is_late = df[status_col].astype(str).str.lower().str.contains("late|delay|backorder", na=False)
            kpis["on_time_delivery_rate"] = 1.0 - is_late.mean()

        # 4. Inventory Health (Enhanced SKU Logic)
        if inventory and pd.api.types.is_numeric_dtype(df[inventory]):
            kpis["avg_inventory_level"] = df[inventory].mean()
            
            # Smart Stockout: If SKU exists, group first (Sum across warehouses)
            if sku:
                stock_by_sku = df.groupby(sku)[inventory].sum()
                kpis["stockout_rate"] = (stock_by_sku <= 0).mean()
            else:
                # Row-based fallback
                kpis["stockout_rate"] = (df[inventory] <= 0).mean()
                
            kpis["target_stockout_rate"] = 0.05

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate KPIs once
        kpis = self.calculate_kpis(df)

        def human_fmt(x, _):
            if abs(x) >= 1_000: return f"{x/1_000:.1f}K"
            return str(int(x))

        order_id = resolve_column(df, "order_id")
        inventory = resolve_column(df, "inventory") or resolve_column(df, "stock")
        supplier = resolve_column(df, "supplier") or resolve_column(df, "vendor")
        category = resolve_column(df, "category") or resolve_column(df, "product")

        delivery_date = resolve_column(df, "delivery_date") or resolve_column(df, "delivered")
        promised_date = resolve_column(df, "promised_date") or resolve_column(df, "expected_date")

        # -------- Visual 1: Order Volume Trend (Smart Aggregation) --------
        if self.has_time_series and order_id:
            p = output_dir / "order_volume_trend.png"
            plt.figure(figsize=(7, 4))

            plot_df = df.copy()
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME")
                    .count() # Counts non-nulls
                    .reset_index()
                )

            plt.plot(plot_df[self.time_col], plot_df[order_id], linewidth=2, color="#9467bd")
            plt.title("Order Fulfillment Trend")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Order processing volume over time"})

        # -------- Visual 2: Delivery Performance --------
        if delivery_date and promised_date:
            p = output_dir / "delivery_performance.png"
            try:
                d_dates = pd.to_datetime(df[delivery_date], errors='coerce')
                p_dates = pd.to_datetime(df[promised_date], errors='coerce')
                
                # Filter to valid rows only
                valid = d_dates.notna() & p_dates.notna()
                if valid.sum() > 0:
                    status = (d_dates[valid] <= p_dates[valid]).value_counts()
                    
                    plt.figure(figsize=(6, 4))
                    # Map boolean to string labels
                    status.rename({True: "On-Time", False: "Delayed"}).plot(
                        kind="bar", color=["#2ca02c", "#d62728"]
                    )
                    plt.title("Delivery Performance")
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    plt.savefig(p)
                    plt.close()
                    visuals.append({"path": p, "caption": "On-time vs delayed deliveries"})
            except Exception:
                pass

        # -------- Visual 3: Inventory by Category (Actionable) --------
        # Replaced Histogram with Category Bar Chart
        if inventory and category and pd.api.types.is_numeric_dtype(df[inventory]):
            p = output_dir / "inventory_by_cat.png"
            
            top_inv = df.groupby(category)[inventory].sum().sort_values(ascending=False).head(7)
            
            plt.figure(figsize=(7, 4))
            top_inv.plot(kind="bar", color="#17becf")
            plt.title("Current Inventory by Category")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Stock distribution across key categories"})

        # -------- Visual 4: Supplier Delay Analysis (High Value) --------
        if supplier and delivery_date and promised_date:
            try:
                d_dates = pd.to_datetime(df[delivery_date], errors='coerce')
                p_dates = pd.to_datetime(df[promised_date], errors='coerce')
                
                # Create boolean 'delayed' column
                df_temp = df.copy()
                df_temp['is_delayed'] = d_dates > p_dates
                
                delay_by_supp = (
                    df_temp.groupby(supplier)['is_delayed']
                    .mean()
                    .sort_values(ascending=False)
                    .head(7)
                )

                if delay_by_supp.max() > 0:
                    p = output_dir / "supplier_delay.png"
                    plt.figure(figsize=(7, 4))
                    delay_by_supp.plot(kind="bar", color="#d62728")
                    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
                    plt.title("Worst Performing Suppliers (Delay Rate)")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(p)
                    plt.close()
                    visuals.append({"path": p, "caption": "Suppliers with highest delay rates"})
            except Exception:
                pass

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        on_time = kpis.get("on_time_delivery_rate")
        stockout = kpis.get("stockout_rate")

        # 1. Delivery Insights
        if on_time is not None:
            if on_time < 0.85:
                insights.append({
                    "level": "RISK",
                    "title": "Critical Delivery Delays",
                    "so_what": f"Only {on_time:.1%} of orders are on time (Target: 95%)."
                })
            elif on_time < 0.95:
                insights.append({
                    "level": "WARNING",
                    "title": "Delivery Standards Slipping",
                    "so_what": f"On-time rate is {on_time:.1%}, slightly below target."
                })

        # 2. Inventory Insights
        if stockout is not None:
            if stockout > 0.10:
                insights.append({
                    "level": "RISK",
                    "title": "High Stockout Rate",
                    "so_what": f"{stockout:.1%} of SKUs are out of stock. Immediate replenishment needed."
                })
            elif stockout > 0.05:
                insights.append({
                    "level": "WARNING",
                    "title": "Inventory Gaps Detected",
                    "so_what": f"Stockout rate is {stockout:.1%}."
                })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Operations Stable",
                "so_what": "Supply chain metrics are performing within normal parameters."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        stockout = kpis.get("stockout_rate")
        on_time = kpis.get("on_time_delivery_rate")

        if stockout is not None and stockout > 0.10:
            recs.append({
                "action": "Initiate emergency replenishment for OOS items",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
        
        if on_time is not None and on_time < 0.85:
            recs.append({
                "action": "Audit lowest-performing suppliers/carriers",
                "priority": "HIGH",
                "timeline": "This Week"
            })

        if not recs:
            recs.append({
                "action": "Maintain current operational processes",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class SupplyChainDomainDetector(BaseDomainDetector):
    domain_name = "supply_chain"

    SUPPLY_CHAIN_TOKENS: Set[str] = {
        "order", "shipment", "delivery",
        "inventory", "stock", "supplier",
        "lead", "logistics"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.SUPPLY_CHAIN_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        return DomainDetectionResult(
            domain="supply_chain",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="supply_chain",
        domain_cls=SupplyChainDomain,
        detector_cls=SupplyChainDomainDetector,
    )
