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
    Marketing-safe time detector.
    """
    candidates = [
        "date", "day", "campaign_date", "report_date",
        "timestamp", "start_date"
    ]

    cols = {c.lower(): c for c in df.columns}

    for key in candidates:
        for low, real in cols.items():
            # FIX 1: Substring match instead of strict equality
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
# MARKETING DOMAIN (v3.1 - POLISHED INTELLIGENCE)
# =====================================================

class MarketingDomain(BaseDomain):
    name = "marketing"
    description = "Marketing & Ad Performance (Campaigns, ROAS, CTR, Channels)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Marketing data needs Campaign signals or Ad Metrics.
        """
        return any(
            resolve_column(df, c) is not None
            for c in [
                "campaign", "campaign_name", "ad_group", "creative",
                "impressions", "clicks", "ctr", "cpc", "cpm",
                "spend", "cost", "ad_spend",
                "source", "medium", "channel"
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

        # 1. Resolve Columns
        spend = resolve_column(df, "spend") or resolve_column(df, "cost") or resolve_column(df, "amount_spent")
        revenue = resolve_column(df, "revenue") or resolve_column(df, "conversion_value")
        impressions = resolve_column(df, "impressions") or resolve_column(df, "imps")
        clicks = resolve_column(df, "clicks")
        conversions = (
            resolve_column(df, "conversions") 
            or resolve_column(df, "results") 
            or resolve_column(df, "leads")
            or resolve_column(df, "purchases")
        )
        campaign = resolve_column(df, "campaign") or resolve_column(df, "campaign_name")

        # 2. Aggregates
        if spend and pd.api.types.is_numeric_dtype(df[spend]):
            kpis["total_spend"] = df[spend].sum()

        if revenue and pd.api.types.is_numeric_dtype(df[revenue]):
            kpis["total_revenue"] = df[revenue].sum()

        if conversions and pd.api.types.is_numeric_dtype(df[conversions]):
            kpis["total_conversions"] = df[conversions].sum()

        if clicks and pd.api.types.is_numeric_dtype(df[clicks]):
            kpis["total_clicks"] = df[clicks].sum()
        
        if impressions and pd.api.types.is_numeric_dtype(df[impressions]):
            kpis["total_impressions"] = df[impressions].sum()

        if campaign:
            kpis["active_campaigns"] = df[campaign].nunique()

        # 3. Efficiency Ratios (The "Golden" Metrics)
        # ROAS (Return on Ad Spend)
        if "total_revenue" in kpis and "total_spend" in kpis:
            kpis["roas"] = _safe_div(kpis["total_revenue"], kpis["total_spend"])
            kpis["target_roas"] = 4.0 # 400% benchmark

        # CPA (Cost Per Acquisition)
        if "total_spend" in kpis and "total_conversions" in kpis:
            kpis["cpa"] = _safe_div(kpis["total_spend"], kpis["total_conversions"])

        # CTR (Click Through Rate)
        if "total_clicks" in kpis and "total_impressions" in kpis:
            kpis["ctr"] = _safe_div(kpis["total_clicks"], kpis["total_impressions"])
            kpis["target_ctr"] = 0.01 # 1% benchmark

        # CPC (Cost Per Click)
        if "total_spend" in kpis and "total_clicks" in kpis:
            kpis["cpc"] = _safe_div(kpis["total_spend"], kpis["total_clicks"])

        return kpis

    # ---------------- VISUALS (MAX 4) ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate KPIs once
        kpis = self.calculate_kpis(df)

        def human_fmt(x, _):
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        spend = resolve_column(df, "spend") or resolve_column(df, "cost")
        revenue = resolve_column(df, "revenue") or resolve_column(df, "conversion_value")
        clicks = resolve_column(df, "clicks")
        conversions = resolve_column(df, "conversions") or resolve_column(df, "leads")
        impressions = resolve_column(df, "impressions")
        
        channel = resolve_column(df, "source") or resolve_column(df, "medium") or resolve_column(df, "platform")
        campaign = resolve_column(df, "campaign") or resolve_column(df, "campaign_name")

        # -------- Visual 1: ROAS / Efficiency Trend --------
        if (
            self.has_time_series 
            and spend and revenue 
            and pd.api.types.is_numeric_dtype(df[spend]) 
            and pd.api.types.is_numeric_dtype(df[revenue])
        ):
            p = output_dir / "roas_trend.png"
            plt.figure(figsize=(7, 4))

            plot_df = df.copy()
            # Aggregate to Month/Week to smooth out noise
            if len(df) > 30:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME")
                    .sum()
                    .reset_index()
                )
            
            # Calculate ROAS for the plot
            s_series = plot_df[spend].replace(0, 1) # Safety
            plot_df["_roas"] = plot_df[revenue] / s_series

            plt.plot(plot_df[self.time_col], plot_df["_roas"], linewidth=2, color="#2ca02c")
            plt.axhline(4.0, color="grey", linestyle="--", alpha=0.5, label="Target (4.0)")
            plt.title("ROAS Trend (Return on Ad Spend)")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Advertising efficiency over time"
            })

        # -------- Visual 2: Channel Performance (Spend vs Revenue) --------
        if channel and spend and revenue and pd.api.types.is_numeric_dtype(df[spend]):
            p = output_dir / "channel_performance.png"
            
            grp = df.groupby(channel)[[spend, revenue]].sum().sort_values(by=spend, ascending=False).head(7)
            
            plt.figure(figsize=(7, 4))
            grp.plot(kind="bar", color=["#d62728", "#2ca02c"]) # Red Spend, Green Revenue
            plt.title("Spend vs Revenue by Channel")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Profitability across different channels"
            })

        # -------- Visual 3: The Marketing Funnel --------
        if impressions and clicks and conversions:
            if all(pd.api.types.is_numeric_dtype(df[c]) for c in [impressions, clicks, conversions]):
                p = output_dir / "marketing_funnel.png"
                
                vals = [df[impressions].sum(), df[clicks].sum(), df[conversions].sum()]
                labels = ["Impressions", "Clicks", "Conversions"]
                
                plt.figure(figsize=(7, 4))
                plt.barh(labels, vals, color="#17becf")
                plt.gca().invert_yaxis() # Top to bottom
                plt.title("Marketing Funnel Volume")
                plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
                
                for i, v in enumerate(vals):
                    plt.text(v, i, " " + human_fmt(v, None), va='center')

                plt.tight_layout()
                plt.savefig(p)
                plt.close()

                visuals.append({
                    "path": p,
                    "caption": "Drop-off from viewing to purchasing"
                })

        # -------- Visual 4: Campaign Scatter (Spend vs CPA) --------
        if (
            campaign and spend and conversions 
            and pd.api.types.is_numeric_dtype(df[spend]) 
            and pd.api.types.is_numeric_dtype(df[conversions])
        ):
            p = output_dir / "campaign_scatter.png"
            
            c_data = df.groupby(campaign)[[spend, conversions]].sum()
            c_data = c_data[c_data[spend] > 0]
            
            c_data["cpa"] = c_data[spend] / c_data[conversions].replace(0, 1)
            c_data = c_data.head(20) # Top 20

            plt.figure(figsize=(7, 4))
            plt.scatter(c_data[spend], c_data["cpa"], alpha=0.6, color="#9467bd")
            plt.title("Campaign Cost vs. CPA")
            plt.xlabel("Total Spend")
            plt.ylabel("Cost Per Acquisition")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Campaign efficiency analysis"
            })

        return visuals[:4]

    # ---------------- ATOMIC INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        roas = kpis.get("roas")
        ctr = kpis.get("ctr")
        cpa = kpis.get("cpa")
        target_ctr = kpis.get("target_ctr", 0.01)

        # 1. ROAS Insight
        if roas is not None:
            if roas < 2.0:
                insights.append({
                    "level": "RISK",
                    "title": "Unprofitable Ad Spend",
                    "so_what": f"ROAS is {roas:.2f}x. You are likely losing money on every ad."
                })
            elif roas < 4.0:
                insights.append({
                    "level": "WARNING",
                    "title": "Low Return on Spend",
                    "so_what": f"ROAS is {roas:.2f}x, below the 4.0x benchmark."
                })
            else:
                insights.append({
                    "level": "INFO",
                    "title": "Healthy Ad Performance",
                    "so_what": f"ROAS is strong at {roas:.2f}x."
                })

        # 2. CTR Insight (FIX 3: Dynamic Threshold)
        if ctr is not None:
            if ctr < target_ctr * 0.5: 
                insights.append({
                    "level": "WARNING",
                    "title": "Low Click-Through Rate",
                    "so_what": f"CTR is {ctr:.2%}. Your ad creatives may be fatigued or irrelevant."
                })

        # 3. CPA Insight
        if cpa is not None:
             insights.append({
                "level": "INFO",
                "title": "Acquisition Cost",
                "so_what": f"You are paying approx {cpa:.2f} to acquire a conversion."
            })

        # === CALL COMPOSITE LAYER (v3.1) ===
        # FIX 2: Guard against small datasets
        if len(df) > 30:
            insights += self.generate_composite_insights(df, kpis)

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Marketing Metrics Stable",
                "so_what": "Spend and conversion metrics are tracking within normal ranges."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (MARKETING v3.1) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Marketing v3 Composite Intelligence Layer.
        Detects Fatigue, Inefficiency, and Scaling Issues.
        """
        insights: List[Dict[str, Any]] = []

        spend = kpis.get("total_spend", 0)
        ctr = kpis.get("ctr")
        roas = kpis.get("roas")
        cpa = kpis.get("cpa")
        imps = kpis.get("total_impressions", 0)

        # 1. Creative Fatigue: High Spend + High Impressions + Low CTR
        if spend > 1000 and imps > 50000 and ctr is not None:
            if ctr < 0.008: # < 0.8%
                insights.append({
                    "level": "RISK",
                    "title": "Creative Fatigue Detected",
                    "so_what": (
                        f"High impressions ({imps:,.0f}) but very low CTR ({ctr:.2%}). "
                        f"Your audience is tired of current ads. Refresh creatives immediately."
                    )
                })

        # 2. Inefficient Scaling: High Spend + Low ROAS
        if spend > 5000 and roas is not None:
            if roas < 2.5:
                insights.append({
                    "level": "WARNING",
                    "title": "Inefficient Scaling",
                    "so_what": (
                        f"You are spending heavily ({spend:,.0f}) but returns are diminishing "
                        f"(ROAS {roas:.2f}x). Tighten targeting or reduce budget."
                    )
                })

        # 3. High Quality Traffic (Good CTR + Good ROAS)
        if ctr is not None and roas is not None:
            if ctr > 0.02 and roas > 4.5:
                insights.append({
                    "level": "INFO",
                    "title": "High-Quality Traffic Source",
                    "so_what": (
                        f"Ads are engaging (CTR {ctr:.1%}) and converting well "
                        f"(ROAS {roas:.2f}x). Recommendation: Scale this segment."
                    )
                })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        roas = kpis.get("roas")
        ctr = kpis.get("ctr")

        if roas is not None and roas < 2.0:
            recs.append({
                "action": "Pause low-ROAS campaigns immediately and audit audience targeting",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
        
        if ctr is not None and ctr < 0.01:
            recs.append({
                "action": "A/B test new ad creatives and headlines to boost CTR",
                "priority": "MEDIUM",
                "timeline": "Next Sprint"
            })

        if not recs:
            recs.append({
                "action": "Scale budget on top-performing campaigns",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR (COLLISION PROOF)
# =====================================================

class MarketingDomainDetector(BaseDomainDetector):
    domain_name = "marketing"

    MARKETING_TOKENS: Set[str] = {
        # Campaign Structure
        "campaign", "ad_group", "creative", "keyword",
        "placement", "ad_set",
        
        # Metrics (Digital)
        "impressions", "clicks", "ctr", "cpc", "cpm",
        "reach", "frequency", "roas", "cpa",
        
        # Channels
        "source", "medium", "referral", "social", "email_marketing"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        
        hits = [c for c in cols if any(t in c for t in self.MARKETING_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        # 🔑 MARKETING DOMINANCE RULE
        mkt_exclusive = any(
            t in c 
            for c in cols 
            for t in {"campaign", "ad_group", "cpc", "ctr", "roas", "impressions"}
        )
        
        if mkt_exclusive:
            confidence = max(confidence, 0.85)

        return DomainDetectionResult(
            domain="marketing",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="marketing",
        domain_cls=MarketingDomain,
        detector_cls=MarketingDomainDetector,
    )
