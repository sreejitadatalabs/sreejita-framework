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
    candidates = ["order_date", "date", "transaction_date", "invoice_date", "created_at"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except:
                continue
    return None


def _compute_rfm(df, customer_col, date_col, sales_col):
    """Calculates Recency, Frequency, Monetary values."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    snapshot = df[date_col].max()

    return df.groupby(customer_col).agg(
        recency=(date_col, lambda x: (snapshot - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(sales_col, "sum"),
    ).reset_index()


def _market_basket_lift(df, order_col, product_col):
    """Identifies top product associations using Lift."""
    # Filter for orders with >1 item
    baskets = df.groupby(order_col)[product_col].apply(set)
    baskets = baskets[baskets.apply(len) > 1]
    
    if baskets.empty: return []

    total = len(baskets)
    item_cnt, pair_cnt = {}, {}

    for items in baskets:
        for i in items: item_cnt[i] = item_cnt.get(i, 0) + 1
        # Sort to ensure (A, B) is same as (B, A)
        items = sorted(list(items))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p = (items[i], items[j])
                pair_cnt[p] = pair_cnt.get(p, 0) + 1

    lifts = []
    for (a, b), cnt in pair_cnt.items():
        if cnt < 3: continue # Noise filter
        pa = item_cnt[a] / total
        pb = item_cnt[b] / total
        lift = (cnt / total) / (pa * pb) if pa * pb > 0 else 0
        lifts.append((a, b, lift, cnt))

    return sorted(lifts, key=lambda x: x[2], reverse=True)


# =====================================================
# RETAIL DOMAIN (UNIVERSAL 10/10)
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Universal Retail Intelligence (Sales, Inventory, Customer, Basket)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)

        # 1. Resolve columns ONCE
        self.cols = {
            "sales": resolve_column(df, "sales") or resolve_column(df, "revenue") or resolve_column(df, "amount"),
            "profit": resolve_column(df, "profit") or resolve_column(df, "margin"),
            "cost": resolve_column(df, "cost") or resolve_column(df, "buying_price"),
            "order": resolve_column(df, "order_id") or resolve_column(df, "invoice"),
            "stock": resolve_column(df, "stock_level") or resolve_column(df, "inventory") or resolve_column(df, "qty_on_hand"),
            "customer": resolve_column(df, "customer_id") or resolve_column(df, "customer_name"),
            "product": resolve_column(df, "product_id") or resolve_column(df, "sku") or resolve_column(df, "item"),
            "category": resolve_column(df, "category") or resolve_column(df, "department")
        }

        # 2. Numeric Safety (Critical Fix)
        for m in ["sales", "profit", "cost", "stock"]:
            if self.cols[m]:
                # Remove currency symbols if present
                if df[self.cols[m]].dtype == object:
                    df[self.cols[m]] = df[self.cols[m]].astype(str).str.replace(r'[$,]', '', regex=True)
                df[self.cols[m]] = pd.to_numeric(df[self.cols[m]], errors='coerce').fillna(0)

        # 3. Date Cleaning
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Sales & AOV
        if c["sales"]:
            kpis["total_sales"] = df[c["sales"]].sum()
            
        if c["order"] and c["sales"]:
            kpis["aov"] = _safe_div(df[c["sales"]].sum(), df[c["order"]].nunique())

        # 2. Inventory Health
        if c["stock"]:
            # Stockout Rate
            stockouts = (df[c["stock"]] <= 0).sum()
            kpis["stockout_rate"] = _safe_div(stockouts, len(df))

        # 3. GMROI (Profit / Avg Inventory Cost)
        if c["profit"] and c["stock"] and c["cost"]:
            # Approx Avg Inventory Value = Sum(Stock * Cost) / Items (Simplified)
            current_inv_val = (df[c["stock"]] * df[c["cost"]]).sum()
            if current_inv_val > 0:
                kpis["gmroi"] = df[c["profit"]].sum() / current_inv_val

        # 4. Customer Churn (RFM)
        if c["customer"] and self.time_col and c["sales"]:
            rfm = _compute_rfm(df, c["customer"], self.time_col, c["sales"])
            # Risk: High value (>median) but inactive (>90 days)
            kpis["rfm_high_value_lost"] = int(
                ((rfm["recency"] > 90) & (rfm["monetary"] > rfm["monetary"].median())).sum()
            )

        # 5. Basket Analysis
        if c["order"] and c["product"]:
            # Only compute if dataset size is manageable (<50k rows) for speed
            if len(df) < 50000:
                lifts = _market_basket_lift(df, c["order"], c["product"])
                if lifts:
                    kpis["top_basket_pair"] = f"{lifts[0][0]} & {lifts[0][1]}"
                    kpis["top_basket_lift"] = lifts[0][2]

        return kpis

    # ---------------- VISUALS (9 CANDIDATES) ----------------

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
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        # 1. Sales Trend
        if self.time_col and c["sales"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample("M")[c["sales"]].sum().plot(ax=ax, color="#1f77b4")
            ax.set_title("Monthly Sales Trend")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "sales_trend.png", "Revenue performance", 0.9, "sales")

        # 2. Top Products
        if c["product"] and c["sales"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["product"])[c["sales"]].sum().nlargest(10).sort_values().plot.barh(ax=ax, color="#ff7f0e")
            ax.set_title("Top 10 Products by Sales")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "top_products.png", "Best sellers", 0.85, "sales")

        # 3. AOV Distribution
        if c["order"] and c["sales"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["order"])[c["sales"]].sum().hist(ax=ax, bins=20, color="green")
            ax.set_title("Order Value Distribution")
            save(fig, "aov_dist.png", "Spending habits", 0.75, "sales")

        # 4. Stockout Rate (Critical Inventory)
        if c["stock"]:
            rate = (df[c["stock"]] <= 0).mean()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie([1-rate, rate], labels=["In Stock", "Out"], autopct="%1.1f%%", colors=["#90ee90", "#ff6347"])
            ax.set_title("Stock Availability")
            # High importance if stockouts are high
            save(fig, "stockout.png", "Inventory risk", 0.95 if rate > 0.1 else 0.6, "inventory")

        # 5. GMROI Distribution (Financial Efficiency)
        if c["profit"] and c["stock"] and c["cost"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            # Row-level GMROI estimate
            gmroi = (df[c["profit"]] / (df[c["stock"]] * df[c["cost"]]).replace(0, np.nan))
            gmroi = gmroi[gmroi.between(-5, 20)] # Filter outliers
            gmroi.hist(ax=ax, bins=20, color="purple")
            ax.set_title("GMROI Distribution")
            save(fig, "gmroi.png", "Inventory profitability", 0.85, "inventory")

        # 6. Inventory Trend
        if self.time_col and c["stock"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample("M")[c["stock"]].mean().plot(ax=ax, color="gray")
            ax.set_title("Avg Inventory Levels")
            save(fig, "inv_trend.png", "Stockholding trend", 0.7, "inventory")

        # 7. RFM Segments
        if c["customer"] and self.time_col and c["sales"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            rfm = _compute_rfm(df, c["customer"], self.time_col, c["sales"])
            rfm["seg"] = pd.qcut(rfm["monetary"], 3, labels=["Low", "Med", "High"])
            rfm["seg"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
            ax.set_title("Customer Value Segments")
            save(fig, "rfm.png", "Customer segmentation", 0.8, "customer")

        # 8. New vs Repeat
        if c["customer"] and c["order"]:
            fig, ax = plt.subplots(figsize=(5, 3))
            freq = df.groupby(c["customer"])[c["order"]].nunique()
            pd.Series({"Repeat": (freq > 1).sum(), "One-time": (freq == 1).sum()}).plot.bar(ax=ax, color="teal")
            ax.set_title("Customer Retention")
            save(fig, "retention.png", "Loyalty mix", 0.75, "customer")

        # 9. Basket Analysis
        if kpis.get("top_basket_lift", 0) > 1.5:
            # We don't have the basket data stored, so we skip plotting to save time
            # Or we could re-run for top 5 rules if needed. 
            pass

        # Sort and Pick Top 4
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS & RISKS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        stockout = kpis.get("stockout_rate", 0)
        churn_risk = kpis.get("rfm_high_value_lost", 0)
        gmroi = kpis.get("gmroi")
        basket_lift = kpis.get("top_basket_lift", 0)

        if stockout > 0.10:
            insights.append({
                "level": "RISK", "title": "High Stockout Rate",
                "so_what": f"{stockout:.1%} of items are out of stock, causing lost sales."
            })

        if churn_risk > 5:
            insights.append({
                "level": "WARNING", "title": "High-Value Churn",
                "so_what": f"{churn_risk} VIP customers have stopped purchasing recently."
            })

        if gmroi is not None and gmroi < 1.0:
            insights.append({
                "level": "WARNING", "title": "Unprofitable Inventory",
                "so_what": f"GMROI is {gmroi:.2f}. For every $1 in inventory, you earn less than $1 profit."
            })

        if basket_lift > 2.0:
            insights.append({
                "level": "INFO", "title": "Bundling Opportunity",
                "so_what": f"Strong affinity detected for {kpis.get('top_basket_pair')} (Lift: {basket_lift:.1f})."
            })

        if not insights:
            insights.append({"level": "INFO", "title": "Retail Operations Stable", "so_what": "Metrics are healthy."})

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        if kpis.get("stockout_rate", 0) > 0.10:
            recs.append({"action": "Review reorder points and expedite replenishment.", "priority": "HIGH"})

        if kpis.get("gmroi", 2) < 1.0:
            recs.append({"action": "Clear out slow-moving inventory (markdowns).", "priority": "HIGH"})

        if kpis.get("rfm_high_value_lost", 0) > 0:
            recs.append({"action": "Launch win-back email campaign for lapsed VIPs.", "priority": "MEDIUM"})

        if kpis.get("top_basket_lift", 0) > 2.0:
            recs.append({"action": f"Create bundle offer for {kpis.get('top_basket_pair')}.", "priority": "LOW"})

        return recs or [{"action": "Monitor sales trends.", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"
    TOKENS: Set[str] = {"sales", "revenue", "order", "sku", "product", "customer", "store", "pos"}

    def detect(self, df) -> DomainDetectionResult:
        cols = {c.lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.TOKENS)]
        confidence = min(len(hits) / 4, 1.0)
        
        # Boost if SKU + Sales exist (Strong retail signal)
        if any("sku" in c for c in hits) and any("sales" in c or "revenue" in c for c in hits):
            confidence = max(confidence, 0.90)

        return DomainDetectionResult("retail", confidence, {"matched_columns": hits})


def register(registry):
    registry.register("retail", RetailDomain, RetailDomainDetector)
