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
    """Supply Chain specific time detection (Order, Ship, Delivery)."""
    candidates = ["order date", "ship date", "delivery date", "date"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[0])
                return c
            except:
                continue
    return None


# =====================================================
# SUPPLY CHAIN DOMAIN (UNIVERSAL 10/10)
# =====================================================

class SupplyChainDomain(BaseDomain):
    name = "supply_chain"
    description = "Universal Supply Chain Intelligence (Inventory, Logistics, Fleet, Sustainability)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        # 1. Resolve columns ONCE here.
        self.cols = {
            # Core
            "order_id": resolve_column(df, "order_id") or resolve_column(df, "shipment_id"),
            "inventory": resolve_column(df, "inventory") or resolve_column(df, "stock_level"),
            "sku": resolve_column(df, "sku") or resolve_column(df, "item_id"),
            "category": resolve_column(df, "category") or resolve_column(df, "product_family"),
            "supplier": resolve_column(df, "supplier") or resolve_column(df, "vendor"),
            "carrier": resolve_column(df, "carrier") or resolve_column(df, "logistics_provider"),
            "cost": resolve_column(df, "cost") or resolve_column(df, "shipping_cost"),
            
            # Fleet & Eco
            "distance": resolve_column(df, "distance") or resolve_column(df, "miles"),
            "weight": resolve_column(df, "weight") or resolve_column(df, "tonnage"),
            "co2": resolve_column(df, "co2") or resolve_column(df, "emissions"),
            
            # Dates
            "order_date": resolve_column(df, "order_date"),
            "ship_date": resolve_column(df, "ship_date"),
            "delivery_date": resolve_column(df, "delivery_date") or resolve_column(df, "actual_delivery"),
            "promised_date": resolve_column(df, "promised_date") or resolve_column(df, "estimated_delivery"),
            
            # Status
            "status": resolve_column(df, "status")
        }

        # 2. Date Cleaning
        for dc in ["order_date", "ship_date", "delivery_date", "promised_date"]:
            if self.cols[dc]:
                df[self.cols[dc]] = pd.to_datetime(df[self.cols[dc]], errors="coerce")

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Volume & OTD
        kpis["total_orders"] = df[c["order_id"]].nunique() if c["order_id"] else len(df)

        if c["delivery_date"] and c["promised_date"]:
            mask = df[c["delivery_date"]].notna() & df[c["promised_date"]].notna()
            if mask.sum() > 0:
                on_time = (df.loc[mask, c["delivery_date"]] <= df.loc[mask, c["promised_date"]])
                kpis["otd_rate"] = on_time.mean()
        elif c["status"]:
            late = df[c["status"]].astype(str).str.lower().str.contains("late|delay", na=False)
            kpis["otd_rate"] = 1.0 - late.mean()

        # 2. Lead Time (Numeric & Variance)
        if c["order_date"] and c["delivery_date"]:
            lead = (df[c["delivery_date"]] - df[c["order_date"]]).dt.days
            valid_lead = lead[(lead >= 0) & (lead < 365)]
            if not valid_lead.empty:
                kpis["avg_lead_time"] = valid_lead.mean()
                kpis["lead_time_cv"] = _safe_div(valid_lead.std(), valid_lead.mean())

        # 3. Inventory
        if c["inventory"]:
            kpis["avg_stock"] = df[c["inventory"]].mean()
            kpis["stockout_rate"] = (df[c["inventory"]] <= 0).mean()

        # 4. Fleet & Cost
        if c["cost"] and kpis["total_orders"]:
            kpis["avg_cost_per_order"] = df[c["cost"]].sum() / kpis["total_orders"]

        if c["cost"] and c["distance"]:
            kpis["cost_per_mile"] = _safe_div(df[c["cost"]].sum(), df[c["distance"]].sum())

        # 5. Sustainability
        if c["co2"]:
            kpis["avg_co2_per_order"] = df[c["co2"]].mean()
        elif c["distance"] and c["weight"]:
            # Estimate: 0.00016 kg CO2 per kg-mile
            est_co2 = df[c["distance"]] * df[c["weight"]] * 0.00016
            kpis["avg_co2_per_order"] = est_co2.mean()

        return kpis

    # ---------------- VISUALS (SMART SELECTION) ----------------

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

        # 1. OTD Trend
        if c["delivery_date"] and c["promised_date"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            temp = df.copy()
            temp['on_time'] = temp[c["delivery_date"]] <= temp[c["promised_date"]]
            temp.set_index(c["promised_date"]).resample('M')['on_time'].mean().plot(ax=ax, color="green")
            ax.set_title("On-Time Delivery Trend")
            ax.set_ylim(0, 1.1)
            save(fig, "otd_trend.png", "Reliability", 0.95 if kpis.get("otd_rate", 1) < 0.9 else 0.8, "service")

        # 2. Lead Time Variance
        if kpis.get("avg_lead_time"):
            fig, ax = plt.subplots(figsize=(6, 4))
            lead = (df[c["delivery_date"]] - df[c["order_date"]]).dt.days
            lead = lead[(lead >= 0) & (lead < 60)]
            lead.hist(ax=ax, bins=20, color="purple", alpha=0.7)
            ax.set_title("Lead Time Distribution")
            imp = 0.9 if kpis.get("lead_time_cv", 0) > 0.5 else 0.75
            save(fig, "lead_time_dist.png", "Process variability", imp, "velocity")

        # 3. Sustainability (CO2)
        if kpis.get("avg_co2_per_order") and c["carrier"]:
            co2_col = c["co2"] if c["co2"] else "est_co2"
            if not c["co2"]: df["est_co2"] = df[c["distance"]] * df[c["weight"]] * 0.00016
            
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["carrier"])[co2_col].mean().plot(kind="bar", ax=ax, color="green")
            ax.set_title("Avg CO2 Emissions per Carrier")
            save(fig, "co2_carrier.png", "Carbon footprint", 0.85, "sustainability")

        # 4. Fleet Efficiency
        if kpis.get("cost_per_mile") and c["carrier"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["carrier"]).apply(lambda x: x[c["cost"]].sum() / x[c["distance"]].sum()).plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Cost per Mile by Carrier")
            save(fig, "cost_mile.png", "Fleet efficiency", 0.82, "cost")

        # 5. Inventory by Category
        if c["inventory"] and c["category"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["category"])[c["inventory"]].sum().nlargest(10).plot(kind="bar", ax=ax)
            ax.set_title("Inventory Levels")
            save(fig, "inventory.png", "Stock distribution", 0.75, "inventory")

        # 6. Supplier Reliability
        if c["supplier"] and c["promised_date"]:
            temp = df.copy()
            temp['late'] = temp[c["delivery_date"]] > temp[c["promised_date"]]
            bad = temp.groupby(c["supplier"])['late'].mean().nlargest(5)
            if bad.max() > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                bad.plot(kind="bar", ax=ax, color="red")
                ax.set_title("Highest Delay Rate by Supplier")
                save(fig, "supplier_risk.png", "Vendor risk", 0.92, "risk")

        # 7. Stockout Rate
        if c["inventory"] and c["category"]:
            temp = df.copy()
            temp['oos'] = temp[c["inventory"]] <= 0
            oos = temp.groupby(c["category"])['oos'].mean().nlargest(5)
            if oos.max() > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                oos.plot(kind="bar", ax=ax, color="orange")
                ax.set_title("Stockout Rate by Category")
                save(fig, "stockouts.png", "Availability risk", 0.9, "inventory")

        # 8. Order Volume
        if self.time_col:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample('M').size().plot(ax=ax)
            ax.set_title("Order Volume")
            save(fig, "volume.png", "Demand trend", 0.8, "planning")

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- COMPOSITE INSIGHTS (THE SMART LAYER) ----------------

    def generate_composite_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        otd = kpis.get("otd_rate", 1.0)
        stockout = kpis.get("stockout_rate", 0.0)
        lead = kpis.get("avg_lead_time", 0)
        inventory_lvl = kpis.get("avg_stock", 0)

        # 1. Fulfillment Crisis (Slow Lead Time + Low OTD)
        if otd < 0.85 and lead > 10:
            insights.append({
                "level": "CRITICAL",
                "title": "Fulfillment Crisis",
                "so_what": f"Delivery is unreliable ({otd:.1%}) AND slow ({lead:.1f} days). Systemic logistics failure."
            })

        # 2. Inventory Imbalance (High Stock + High Stockouts)
        # Assuming 'High Stock' is relative, checking if avg > 0
        if inventory_lvl > 0 and stockout > 0.15:
            insights.append({
                "level": "RISK",
                "title": "Inventory Imbalance",
                "so_what": f"You have stock, but {stockout:.1%} of items are OOS. Mismatched demand/supply."
            })

        return insights

    # ---------------- ATOMIC INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Composite First
        insights = self.generate_composite_insights(df, kpis)
        titles = [i["title"] for i in insights]

        # Atomic Fallbacks
        cv = kpis.get("lead_time_cv", 0)
        co2 = kpis.get("avg_co2_per_order")
        cpm = kpis.get("cost_per_mile")
        otd = kpis.get("otd_rate", 1.0)

        # Variance
        if cv > 0.5:
            insights.append({
                "level": "RISK", "title": "Unstable Lead Times",
                "so_what": f"High variability (CV: {cv:.2f}). Planning buffers likely insufficient."
            })

        # Sustainability
        if co2 is not None and co2 > 50:
            insights.append({
                "level": "WARNING", "title": "High Carbon Footprint",
                "so_what": f"Avg CO2 per order is {co2:.1f}kg."
            })

        # Cost
        if cpm is not None and cpm > 3.0:
            insights.append({
                "level": "WARNING", "title": "High Fleet Costs",
                "so_what": f"Cost per mile is ${cpm:.2f}."
            })

        # OTD (if not covered by Crisis)
        if otd < 0.90 and "Fulfillment Crisis" not in titles:
            insights.append({
                "level": "WARNING", "title": "Delivery Delays", 
                "so_what": f"OTD is {otd:.1%}, below target."
            })

        if not insights:
            insights.append({"level": "INFO", "title": "Supply Chain Stable", "so_what": "Operations nominal."})

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        # Composite Actions
        if "Fulfillment Crisis" in titles:
            recs.append({"action": "Audit carrier SLAs and switch underperformers immediately.", "priority": "HIGH"})
        
        if "Inventory Imbalance" in titles:
            recs.append({"action": "Markdown overstock and expedite OOS items.", "priority": "HIGH"})

        # Atomic Actions
        if kpis.get("lead_time_cv", 0) > 0.5:
            recs.append({"action": "Increase safety stock buffers.", "priority": "MEDIUM"})

        if kpis.get("avg_co2_per_order"):
            recs.append({"action": "Review route optimization for carbon reduction.", "priority": "MEDIUM"})

        if not recs:
            recs.append({"action": "Monitor metrics.", "priority": "LOW"})

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class SupplyChainDomainDetector(BaseDomainDetector):
    domain_name = "supply_chain"
    TOKENS = {"inventory", "stock", "shipping", "delivery", "carrier", "supplier", "freight", "logistics"}

    def detect(self, df) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # Boost if Inventory AND Delivery exist
        cols = str(df.columns).lower()
        if "stock" in cols and "delivery" in cols:
            confidence = max(confidence, 0.95)
            
        return DomainDetectionResult("supply_chain", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("supply_chain", SupplyChainDomain, SupplyChainDomainDetector)
