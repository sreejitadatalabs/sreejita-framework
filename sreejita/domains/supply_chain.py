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
# SUPPLY CHAIN / OPERATIONS DOMAIN (v3.1 - ENTERPRISE)
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
            df = _prepare_time_series(df, self.time_col)
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
        else:
            kpis["order_volume"] = len(df) 

        # 2. Lead Time (Actual Calculation)
        if order_date and delivery_date:
            try:
                start = pd.to_datetime(df[order_date], errors='coerce')
                end = pd.to_datetime(df[delivery_date], errors='coerce')
                lead_time = (end - start).dt.days
                
                # Filter out negative or crazy outliers (> 365 days)
                valid_leads = lead_time[(lead_time >= 0) & (lead_time < 365)]
                
                if not valid_leads.empty:
                    kpis["avg_lead_time_days"] = valid_leads.mean()
            except Exception:
                pass

        # 3. On-Time Delivery (The Hybrid Approach)
        kpis["target_otd_rate"] = 0.95 # Benchmark

        if delivery_date and promised_date:
            try:
                delivered = pd.to_datetime(df[delivery_date], errors='coerce')
                promised = pd.to_datetime(df[promised_date], errors='coerce')
                mask = delivered.notna() & promised.notna()
                if mask.sum() > 0:
                    on_time = (delivered[mask] <= promised[mask]).mean()
                    kpis["on_time_delivery_rate"] = on_time
            except Exception:
                pass
        
        # Fallback: Status Text Check
        if "on_time_delivery_rate" not in kpis and status_col:
            is_late = df[status_col].astype(str).str.lower().str.contains("late|delay|backorder|fail", na=False)
            kpis["on_time_delivery_rate"] = 1.0 - is_late.mean()

        # 4. Inventory Health (Enhanced SKU Logic)
        if inventory and pd.api.types.is_numeric_dtype(df[inventory]):
            kpis["avg_inventory_level"] = df[inventory].mean()
            
            # Smart Stockout: If SKU exists, group first (Sum across warehouses)
            if sku:
                stock_by_sku = df.groupby(sku)[inventory].sum()
                kpis["stockout_rate"] = (stock_by_sku <= 0).mean()
            else:
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

        inventory = resolve_column(df, "inventory") or resolve_column(df, "stock")
        supplier = resolve_column(df, "supplier") or resolve_column(df, "vendor")
        category = resolve_column(df, "category") or resolve_column(df, "product")
        carrier = resolve_column(df, "carrier") or resolve_column(df, "shipping_carriers")

        delivery_date = resolve_column(df, "delivery_date") or resolve_column(df, "delivered")
        promised_date = resolve_column(df, "promised_date") or resolve_column(df, "expected_date")

        # -------- Visual 1: Order Volume Trend --------
        if self.has_time_series:
            p = output_dir / "order_volume_trend.png"
            plt.figure(figsize=(7, 4))

            plot_df = df.copy()
            # Smart Aggregation
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME")
                    .size()
                    .reset_index(name="count")
                )
                plt.plot(plot_df[self.time_col], plot_df["count"], linewidth=2, color="#9467bd")
            else:
                # Group by exact date for small datasets
                dates = pd.to_datetime(df[self.time_col]).dt.date
                dates.value_counts().sort_index().plot(linewidth=2, color="#9467bd")

            plt.title("Order Fulfillment Trend")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Order processing volume over time"})

        # -------- Visual 2: Inventory by Category --------
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

        # -------- Visual 3: Supplier / Carrier Performance --------
        # Priority 1: Carrier Cost/Usage
        if carrier:
            p = output_dir / "carrier_usage.png"
            c_counts = df[carrier].value_counts().head(7)
            
            plt.figure(figsize=(7, 4))
            c_counts.plot(kind="barh", color="#bcbd22")
            plt.title("Top Shipping Carriers by Volume")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            visuals.append({"path": p, "caption": "Utilization of logistics providers"})
            
        # Priority 2: Supplier Delay (if dates available)
        elif supplier and delivery_date and promised_date:
            try:
                d_dates = pd.to_datetime(df[delivery_date], errors='coerce')
                p_dates = pd.to_datetime(df[promised_date], errors='coerce')
                
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

    # ---------------- ATOMIC INSIGHTS ----------------

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
        
        # === CALL COMPOSITE LAYER (v3.1) ===
        # Guard: Only call composite insights if dataset is significant enough
        if len(df) > 30:
            insights += self.generate_composite_insights(df, kpis)

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Operations Stable",
                "so_what": "Supply chain metrics are performing within normal parameters."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (SC v3.1) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Supply Chain v3.1 Composite Intelligence Layer.
        Detects Inventory Imbalance, Fulfillment Bottlenecks.
        """
        insights: List[Dict[str, Any]] = []

        otd = kpis.get("on_time_delivery_rate")
        stockout = kpis.get("stockout_rate")
        lead_time = kpis.get("avg_lead_time_days")
        inventory_lvl = kpis.get("avg_inventory_level")

        # 1. Inventory Imbalance (High Stock + High Stockouts)
        # Means we are holding a lot of "Dead Stock" while popular items are missing.
        if inventory_lvl is not None and stockout is not None:
            if inventory_lvl > 0 and stockout > 0.10:
                 insights.append({
                    "level": "WARNING",
                    "title": "Inventory Imbalance Detected",
                    "so_what": (
                        f"Stockout rate is high ({stockout:.1%}) despite holding significant inventory. "
                        f"You are likely overstocked on slow-movers and understocked on key items."
                    )
                })

        # 2. Fulfillment Bottleneck (Low OTD + Slow Lead Time)
        if otd is not None and lead_time is not None:
            if otd < 0.85 and lead_time > 10: # >10 days avg lead time
                 insights.append({
                    "level": "RISK",
                    "title": "Severe Fulfillment Bottleneck",
                    "so_what": (
                        f"Delivery reliability is low ({otd:.1%}) with extended lead times "
                        f"({lead_time:.1f} days). Logistics or carrier performance requires audit."
                    )
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
# DOMAIN DETECTOR (COLLISION PROOF)
# =====================================================

class SupplyChainDomainDetector(BaseDomainDetector):
    domain_name = "supply_chain"

    # Expanded tokens including upstream keywords to distinguish from Retail
    SUPPLY_CHAIN_TOKENS: Set[str] = {
        # Inventory & Stock
        "inventory", "stock", "onhand", "warehouse",

        # Logistics & Fulfillment
        "shipment", "delivery", "dispatch", "carrier",
        "freight", "logistics", "route", "transport", "transportation",

        # Performance & Timing
        "lead_time", "cycle_time", "turnaround",
        "delay", "on_time", "backorder",

        # Operations
        "sku", "quantity", "units", "volume",
        "supplier", "vendor", "procurement",
        "manufacturing", "production", "defect"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        
        hits = [c for c in cols if any(t in c for t in self.SUPPLY_CHAIN_TOKENS)]
        
        # Base confidence calculation
        confidence = min(len(hits) / 3, 1.0)

        # 🔑 SUPPLY CHAIN DOMINANCE RULE
        # These words strongly imply upstream/logistics operations, overruling generic "Sales" signals
        sc_exclusive = any(
            t in c
            for c in cols
            for t in {
                "inventory", "stock", "warehouse",
                "delivery", "shipment", "lead_time",
                "carrier", "supplier", "freight",
                "manufacturing", "production"
            }
        )

        if sc_exclusive:
            confidence = max(confidence, 0.85)

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
``` [attachment_0](attachment)

