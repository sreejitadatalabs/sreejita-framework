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
    """Marketing-safe time detector."""
    candidates = ["date", "day", "campaign_date", "timestamp", "start_date"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except:
                continue
    return None


# =====================================================
# MARKETING DOMAIN (UNIVERSAL 10/10)
# =====================================================

class MarketingDomain(BaseDomain):
    name = "marketing"
    description = "Universal Marketing Intelligence (Campaigns, ROAS, Channels, Creatives)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        # 1. Resolve columns ONCE.
        self.cols = {
            # Metrics
            "spend": resolve_column(df, "spend") or resolve_column(df, "cost") or resolve_column(df, "amount_spent"),
            "revenue": resolve_column(df, "revenue") or resolve_column(df, "conversion_value") or resolve_column(df, "return"),
            "impressions": resolve_column(df, "impressions") or resolve_column(df, "imps") or resolve_column(df, "views"),
            "clicks": resolve_column(df, "clicks") or resolve_column(df, "link_clicks"),
            "conversions": resolve_column(df, "conversions") or resolve_column(df, "purchases") or resolve_column(df, "results"),
            
            # Dimensions
            "campaign": resolve_column(df, "campaign") or resolve_column(df, "campaign_name"),
            "channel": resolve_column(df, "source") or resolve_column(df, "platform") or resolve_column(df, "medium"),
            "ad": resolve_column(df, "ad_name") or resolve_column(df, "creative"),
            "keyword": resolve_column(df, "keyword") or resolve_column(df, "term")
        }

        # 2. Date Cleaning
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        # 3. Numeric Safety
        for m in ["spend", "revenue", "impressions", "clicks", "conversions"]:
            if self.cols[m]:
                df[self.cols[m]] = pd.to_numeric(df[self.cols[m]], errors='coerce').fillna(0)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Aggregates
        if c["spend"]: kpis["total_spend"] = df[c["spend"]].sum()
        if c["revenue"]: kpis["total_revenue"] = df[c["revenue"]].sum()
        if c["conversions"]: kpis["total_conversions"] = df[c["conversions"]].sum()
        if c["clicks"]: kpis["total_clicks"] = df[c["clicks"]].sum()
        if c["impressions"]: kpis["total_impressions"] = df[c["impressions"]].sum()

        # 2. Efficiency Ratios
        if kpis.get("total_spend"):
            # ROAS
            if kpis.get("total_revenue"):
                kpis["roas"] = _safe_div(kpis["total_revenue"], kpis["total_spend"])
            
            # CPA (Cost Per Acquisition)
            if kpis.get("total_conversions"):
                kpis["cpa"] = _safe_div(kpis["total_spend"], kpis["total_conversions"])
            
            # CPM (Cost Per Mille)
            if kpis.get("total_impressions"):
                kpis["cpm"] = _safe_div(kpis["total_spend"], kpis["total_impressions"]) * 1000

        # CTR (Click Through Rate)
        if kpis.get("total_clicks") and kpis.get("total_impressions"):
            kpis["ctr"] = _safe_div(kpis["total_clicks"], kpis["total_impressions"])

        # CPC (Cost Per Click)
        if kpis.get("total_spend") and kpis.get("total_clicks"):
            kpis["cpc"] = _safe_div(kpis["total_spend"], kpis["total_clicks"])

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
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        # 1. ROAS Trend (Efficiency)
        if self.time_col and c["spend"] and c["revenue"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            temp = df.set_index(self.time_col).resample('M').sum()
            roas_series = temp[c["revenue"]] / temp[c["spend"]].replace(0, 1)
            roas_series.plot(ax=ax, color="#2ca02c", linewidth=2)
            ax.axhline(4.0, color="gray", linestyle="--", label="Target")
            ax.set_title("ROAS Trend")
            # High priority if ROAS is low/dropping
            imp = 0.95 if kpis.get("roas", 5) < 3.0 else 0.85
            save(fig, "roas_trend.png", "Efficiency over time", imp, "performance")

        # 2. Channel Performance (Bar)
        if c["channel"] and c["spend"] and c["revenue"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["channel"])[[c["spend"], c["revenue"]]].sum().nlargest(7, c["spend"]).plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"])
            ax.set_title("Spend vs Revenue by Channel")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "channel_perf.png", "Channel ROI", 0.9, "strategy")

        # 3. Marketing Funnel
        if c["impressions"] and c["clicks"] and c["conversions"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            vals = [df[c["impressions"]].sum(), df[c["clicks"]].sum(), df[c["conversions"]].sum()]
            ax.barh(["Impressions", "Clicks", "Conversions"], vals, color="#17becf")
            ax.invert_yaxis()
            ax.set_title("Marketing Funnel")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "funnel.png", "Conversion flow", 0.8, "funnel")

        # 4. Campaign Scatter (Cost vs CPA)
        if c["campaign"] and c["spend"] and c["conversions"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            agg = df.groupby(c["campaign"]).agg({c["spend"]: "sum", c["conversions"]: "sum"})
            agg = agg[agg[c["spend"]] > 0] # Filter zero spend
            agg["cpa"] = agg[c["spend"]] / agg[c["conversions"]].replace(0, 1)
            
            ax.scatter(agg[c["spend"]], agg["cpa"], alpha=0.6, color="#9467bd")
            ax.set_title("Campaign Cost vs CPA")
            ax.set_xlabel("Total Spend")
            ax.set_ylabel("CPA")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "campaign_cpa.png", "Cost efficiency matrix", 0.85, "optimization")

        # 5. CTR Distribution (New!)
        if c["ctr"]: # If pre-calculated
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["ctr"]].hist(ax=ax, bins=20, color="orange")
            ax.set_title("Ad CTR Distribution")
            save(fig, "ctr_dist.png", "Creative engagement", 0.75, "creative")
        elif c["clicks"] and c["impressions"]:
            # Calculate row-level CTR safely
            fig, ax = plt.subplots(figsize=(6, 4))
            mask = df[c["impressions"]] > 100 # Filter low volume noise
            (df.loc[mask, c["clicks"]] / df.loc[mask, c["impressions"]]).hist(ax=ax, bins=20, color="orange")
            ax.set_title("Ad CTR Distribution")
            # High importance if CTR is extremely low
            imp = 0.9 if kpis.get("ctr", 0.05) < 0.01 else 0.75
            save(fig, "ctr_dist.png", "Creative engagement", imp, "creative")

        # 6. CPM Trend (Market Cost) (New!)
        if self.time_col and c["spend"] and c["impressions"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            temp = df.set_index(self.time_col).resample('M').sum()
            cpm = (temp[c["spend"]] / temp[c["impressions"]].replace(0, 1)) * 1000
            cpm.plot(ax=ax, color="brown")
            ax.set_title("CPM Trend (Cost per 1k Views)")
            save(fig, "cpm_trend.png", "Market price fluctuation", 0.7, "cost")

        # 7. Day of Week Performance (New!)
        if self.time_col and c["conversions"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(df[self.time_col].dt.day_name())[c["conversions"]].sum().reindex(
                ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            ).plot(kind="bar", ax=ax, color="teal")
            ax.set_title("Conversions by Day")
            save(fig, "dow_conv.png", "Timing optimization", 0.65, "tactical")

        # 8. Clicks vs Impressions (Relevance) (New!)
        if c["clicks"] and c["impressions"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df[c["impressions"]], df[c["clicks"]], alpha=0.5, color="gray")
            ax.set_title("Impressions vs Clicks")
            ax.set_xlabel("Impressions")
            ax.set_ylabel("Clicks")
            save(fig, "imps_clicks.png", "Ad relevance correlation", 0.7, "creative")

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        roas = kpis.get("roas")
        ctr = kpis.get("ctr")
        spend = kpis.get("total_spend", 0)
        imps = kpis.get("total_impressions", 0)

        # Composite 1: Creative Fatigue (High Spend, High Imps, Low CTR)
        if spend > 1000 and imps > 50000 and ctr is not None:
            if ctr < 0.008: 
                insights.append({
                    "level": "RISK", "title": "Creative Fatigue",
                    "so_what": f"Ads have high reach ({imps:,.0f}) but low engagement ({ctr:.2%}). Audience is saturated."
                })

        # Composite 2: Inefficient Scaling (High Spend, Low ROAS)
        if spend > 5000 and roas is not None:
            if roas < 2.0:
                insights.append({
                    "level": "WARNING", "title": "Inefficient Scaling",
                    "so_what": f"High spend ({spend:,.0f}) is yielding poor returns (ROAS {roas:.2f}x)."
                })

        # Atomic Fallbacks
        if roas is not None and roas < 3.0 and not any("Scaling" in i["title"] for i in insights):
            insights.append({
                "level": "WARNING", "title": "Low ROAS", 
                "so_what": f"Return on Ad Spend is {roas:.2f}x, below target."
            })

        if ctr is not None and ctr < 0.01 and not any("Fatigue" in i["title"] for i in insights):
            insights.append({
                "level": "WARNING", "title": "Low CTR", 
                "so_what": f"Click-through rate is {ctr:.2%}. Creatives may need refresh."
            })

        if not insights:
            insights.append({"level": "INFO", "title": "Campaigns Stable", "so_what": "Performance metrics healthy."})

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if "Creative Fatigue" in titles:
            recs.append({"action": "Rotate in new ad creatives immediately.", "priority": "HIGH"})
        
        if "Inefficient Scaling" in titles:
            recs.append({"action": "Cut budget on low-ROAS ad sets.", "priority": "HIGH"})

        if kpis.get("ctr", 0) < 0.01:
            recs.append({"action": "A/B test new headlines to boost CTR.", "priority": "MEDIUM"})

        return recs or [{"action": "Scale top performing ads.", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class MarketingDomainDetector(BaseDomainDetector):
    domain_name = "marketing"
    TOKENS = {"campaign", "ad_group", "cpc", "ctr", "roas", "impressions", "clicks", "spend", "cost"}

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # Boost if Spend + Impressions exist (Distinct from Sales/Retail)
        if any("spend" in c or "cost" in c for c in hits) and any("impressions" in c for c in hits):
            confidence = max(confidence, 0.95)
            
        return DomainDetectionResult("marketing", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("marketing", MarketingDomain, MarketingDomainDetector)
