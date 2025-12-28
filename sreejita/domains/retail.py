import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

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
    candidates = [
        "order_date", "date", "transaction_date",
        "invoice_date", "created_date", "timestamp"
    ]
    for c in df.columns:
        for k in candidates:
            if k in c.lower():
                try:
                    pd.to_datetime(df[c].dropna().iloc[0])
                    return c
                except Exception:
                    pass
    return None


def _compute_rfm(df, customer_col, date_col, sales_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    snapshot = df[date_col].max()

    return (
        df.groupby(customer_col)
        .agg(
            recency=(date_col, lambda x: (snapshot - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(sales_col, "sum"),
        )
        .reset_index()
    )


def _market_basket_lift(df, order_col, product_col):
    baskets = df.groupby(order_col)[product_col].apply(set)
    baskets = baskets[baskets.apply(len) > 1]
    if baskets.empty:
        return []

    total = len(baskets)
    item_cnt, pair_cnt = {}, {}

    for items in baskets:
        for i in items:
            item_cnt[i] = item_cnt.get(i, 0) + 1
        items = sorted(items)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p = (items[i], items[j])
                pair_cnt[p] = pair_cnt.get(p, 0) + 1

    lifts = []
    for (a, b), cnt in pair_cnt.items():
        if cnt < 3:
            continue
        pa = item_cnt[a] / total
        pb = item_cnt[b] / total
        lift = (cnt / total) / (pa * pb) if pa * pb > 0 else 0
        lifts.append((a, b, lift, cnt))

    return sorted(lifts, key=lambda x: x[2], reverse=True)


# =====================================================
# RETAIL DOMAIN — UNIVERSAL (FINAL)
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Universal Retail Intelligence (Sales, Inventory, Customer, Basket)"

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)

        self.cols = {
            "sales": resolve_column(df, "sales") or resolve_column(df, "revenue"),
            "profit": resolve_column(df, "profit"),
            "cost": resolve_column(df, "cost"),
            "order": resolve_column(df, "order_id"),
            "stock": resolve_column(df, "stock_level"),
            "customer": resolve_column(df, "customer_id"),
            "product": resolve_column(df, "product_id") or resolve_column(df, "sku"),
        }

        if self.cols["sales"]:
            df[self.cols["sales"]] = df[self.cols["sales"]].fillna(0)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}
        c = self.cols

        if c["sales"]:
            kpis["total_sales"] = df[c["sales"]].sum()

        if c["order"] and c["sales"]:
            kpis["aov"] = _safe_div(df[c["sales"]].sum(), df[c["order"]].nunique())

        if c["stock"]:
            kpis["stockout_rate"] = (df[c["stock"]] <= 0).mean()

        if c["profit"] and c["stock"] and c["cost"]:
            inv_val = (df[c["stock"]] * df[c["cost"]]).sum()
            kpis["gmroi"] = _safe_div(df[c["profit"]].sum(), inv_val)

        if c["customer"] and self.time_col and c["sales"]:
            rfm = _compute_rfm(df, c["customer"], self.time_col, c["sales"])
            kpis["rfm_high_value_lost"] = int(
                ((rfm["recency"] > 90) & (rfm["monetary"] > rfm["monetary"].median())).sum()
            )

        if c["order"] and c["product"]:
            lifts = _market_basket_lift(df, c["order"], c["product"])
            if lifts:
                kpis["top_basket_pair"] = f"{lifts[0][0]} & {lifts[0][1]}"
                kpis["top_basket_lift"] = lifts[0][2]

        return kpis

    # ---------------- INSIGHTS & RISKS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        # --- INVENTORY RISKS ---
        if kpis.get("stockout_rate", 0) > 0.10:
            insights.append({
                "level": "RISK",
                "title": "High Stockout Rate",
                "so_what": f"{kpis['stockout_rate']:.1%} of items are unavailable, causing lost sales.",
                "category": "inventory",
            })

        if kpis.get("gmroi") is not None and kpis["gmroi"] < 1.0:
            insights.append({
                "level": "WARNING",
                "title": "Unprofitable Inventory",
                "so_what": "Inventory costs exceed profit generated (GMROI < 1).",
                "category": "inventory",
            })

        # --- CUSTOMER RISKS ---
        if kpis.get("rfm_high_value_lost", 0) > 5:
            insights.append({
                "level": "WARNING",
                "title": "High-Value Customer Churn",
                "so_what": f"{kpis['rfm_high_value_lost']} valuable customers have stopped buying.",
                "category": "customer",
            })

        # --- GROWTH OPPORTUNITIES ---
        if kpis.get("top_basket_lift", 0) > 2.0:
            insights.append({
                "level": "INFO",
                "title": "Product Bundling Opportunity",
                "so_what": f"Strong affinity detected between {kpis.get('top_basket_pair')}.",
                "category": "basket",
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Retail Performance Stable",
                "so_what": "Sales, inventory, and customer metrics are within expected ranges.",
                "category": "overall",
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if kpis.get("stockout_rate", 0) > 0.10:
            recs.append({
                "action": "Review reorder points and supplier lead times to reduce stockouts.",
                "priority": "HIGH",
            })

        if kpis.get("gmroi") is not None and kpis["gmroi"] < 1.0:
            recs.append({
                "action": "Mark down or discontinue low-performing inventory to free working capital.",
                "priority": "HIGH",
            })

        if kpis.get("rfm_high_value_lost", 0) > 0:
            recs.append({
                "action": "Launch targeted win-back campaigns for high-value customers.",
                "priority": "MEDIUM",
            })

        if kpis.get("top_basket_lift", 0) > 2.0:
            recs.append({
                "action": f"Create bundles or co-promotions for {kpis.get('top_basket_pair')}.",
                "priority": "LOW",
            })

        if not recs:
            recs.append({
                "action": "Continue monitoring retail KPIs and customer behavior.",
                "priority": "LOW",
            })

        return recs

    # ---------------- VISUALS (9 GENERATED, RANKED) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols

        # 1. Sales Trend
        if self.time_col and c["sales"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample("M")[c["sales"]].sum().plot(ax=ax)
            ax.set_title("Monthly Sales Trend")
            p = output_dir / "sales_trend.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "Sales trend", "category": "sales", "importance": 0.90})

        # 2. Sales by Product
        if c["product"] and c["sales"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["product"])[c["sales"]].sum().nlargest(10).sort_values().plot.barh(ax=ax)
            ax.set_title("Top Products by Sales")
            p = output_dir / "sales_by_product.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "Sales by product", "category": "sales", "importance": 0.85})

        # 3. AOV Distribution
        if c["order"] and c["sales"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["order"])[c["sales"]].sum().hist(ax=ax, bins=20)
            ax.set_title("Order Value Distribution")
            p = output_dir / "aov_dist.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "AOV distribution", "category": "sales", "importance": 0.75})

        # 4. Stockout Rate
        if c["stock"]:
            rate = (df[c["stock"]] <= 0).mean()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie([1-rate, rate], labels=["In Stock", "Out"], autopct="%1.1f%%")
            ax.set_title("Stock Availability")
            p = output_dir / "stockout.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "Stockout rate", "category": "inventory", "importance": 0.95 if rate > 0.1 else 0.6})

        # 5. GMROI Distribution
        if c["profit"] and c["stock"] and c["cost"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            gmroi = (df[c["profit"]] / (df[c["stock"]] * df[c["cost"]]).replace(0, np.nan))
            gmroi.hist(ax=ax, bins=20)
            ax.set_title("GMROI Distribution")
            p = output_dir / "gmroi.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "GMROI distribution", "category": "inventory", "importance": 0.85})

        # 6. Inventory Aging (proxy via time)
        if self.time_col and c["stock"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample("M")[c["stock"]].mean().plot(ax=ax)
            ax.set_title("Average Inventory Over Time")
            p = output_dir / "inventory_trend.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "Inventory aging trend", "category": "inventory", "importance": 0.70})

        # 7. RFM Segments
        if c["customer"] and self.time_col and c["sales"]:
            rfm = _compute_rfm(df, c["customer"], self.time_col, c["sales"])
            rfm["seg"] = pd.qcut(rfm["monetary"], 3, labels=["Bronze", "Silver", "Gold"])
            fig, ax = plt.subplots(figsize=(6, 4))
            rfm["seg"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%")
            ax.set_title("Customer Value Segments")
            p = output_dir / "rfm.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "Customer segments", "category": "customer", "importance": 0.80})

        # 8. Repeat vs New Customers
        if c["customer"] and c["order"]:
            fig, ax = plt.subplots(figsize=(5, 3))
            freq = df.groupby(c["customer"])[c["order"]].nunique()
            pd.Series({"Repeat": (freq > 1).sum(), "New": (freq == 1).sum()}).plot.bar(ax=ax)
            ax.set_title("Repeat vs New Customers")
            p = output_dir / "repeat_new.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            visuals.append({"path": str(p), "caption": "Repeat vs new customers", "category": "customer", "importance": 0.75})

        # 9. Market Basket Lift
        if c["order"] and c["product"]:
            lifts = _market_basket_lift(df, c["order"], c["product"])[:5]
            if lifts:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh([f"{a}+{b}" for a, b, _, _ in lifts], [l for _, _, l, _ in lifts])
                ax.set_title("Top Product Bundles (Lift)")
                p = output_dir / "basket.png"
                fig.savefig(p, bbox_inches="tight"); plt.close(fig)
                visuals.append({"path": str(p), "caption": "Product bundles", "category": "basket", "importance": 0.70})

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"
    TOKENS: Set[str] = {"sales", "revenue", "order", "sku", "product", "customer", "store"}

    def detect(self, df) -> DomainDetectionResult:
        cols = {c.lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.TOKENS)]
        confidence = min(len(hits) / 4, 1.0)
        if any("sku" in c for c in cols) and any("sales" in c for c in cols):
            confidence = max(confidence, 0.90)

        return DomainDetectionResult("retail", confidence, {"matched_columns": hits})


def register(registry):
    registry.register("retail", RetailDomain, RetailDomainDetector)
