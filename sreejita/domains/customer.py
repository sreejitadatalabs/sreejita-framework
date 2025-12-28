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


# =====================================================
# CUSTOMER DOMAIN (UNIVERSAL 10/10)
# =====================================================

class CustomerDomain(BaseDomain):
    name = "customer"
    description = "Universal Customer Intelligence (CX, Loyalty, Support, Churn)"

    # ---------------- PREPROCESS (SAFETY LAYER) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Resolve Columns
        self.cols = {
            "customer": resolve_column(df, "customer_id") or resolve_column(df, "customer"),
            "nps": resolve_column(df, "nps") or resolve_column(df, "net_promoter_score"),
            "csat": resolve_column(df, "csat") or resolve_column(df, "satisfaction"),
            "ces": resolve_column(df, "ces") or resolve_column(df, "effort_score"),
            "churn": resolve_column(df, "churn") or resolve_column(df, "churned"),
            "ticket": resolve_column(df, "ticket_id") or resolve_column(df, "case_id"),
            "frt": resolve_column(df, "first_response_time") or resolve_column(df, "frt"),
            "art": resolve_column(df, "avg_resolution_time") or resolve_column(df, "resolution_time"),
            "fcr": resolve_column(df, "fcr") or resolve_column(df, "first_contact_resolution"),
            "sentiment": resolve_column(df, "sentiment_score") or resolve_column(df, "sentiment"),
        }

        # 2. Force Numeric Types (Safety Fix)
        numeric_cols = ["nps", "csat", "ces", "frt", "art", "sentiment", "fcr", "churn"]
        for key in numeric_cols:
            col_name = self.cols.get(key)
            if col_name:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        if c["customer"]:
            kpis["customer_count"] = df[c["customer"]].nunique()

        if c["nps"]:
            kpis["avg_nps"] = df[c["nps"]].mean()

        if c["csat"]:
            kpis["avg_csat"] = df[c["csat"]].mean()

        if c["ces"]:
            kpis["avg_ces"] = df[c["ces"]].mean()

        if c["churn"]:
            kpis["churn_rate"] = df[c["churn"]].mean()

        if c["ticket"]:
            kpis["ticket_volume"] = df[c["ticket"]].nunique()

        if c["frt"]:
            kpis["avg_first_response_time"] = df[c["frt"]].mean()

        if c["art"]:
            kpis["avg_resolution_time"] = df[c["art"]].mean()

        if c["fcr"]:
            kpis["first_contact_resolution_rate"] = df[c["fcr"]].mean()

        if c["sentiment"]:
            kpis["avg_sentiment"] = df[c["sentiment"]].mean()

        return kpis

    # ---------------- VISUALS (8 CANDIDATES, TOP 4 SELECTED) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
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

        # 1. NPS Distribution
        if c["nps"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["nps"]].hist(ax=ax, bins=10, color="#1f77b4")
            ax.set_title("NPS Distribution")
            # Logic: Negative NPS is high priority issue
            save(fig, "nps_dist.png", "Loyalty distribution", 
                 0.95 if kpis.get("avg_nps", 0) < 0 else 0.8, "loyalty")

        # 2. CSAT Distribution
        if c["csat"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["csat"]].hist(ax=ax, bins=5, color="green")
            ax.set_title("CSAT Scores")
            save(fig, "csat_dist.png", "Customer satisfaction", 0.75, "satisfaction")

        # 3. Churn Rate
        if "churn_rate" in kpis:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Churn Rate"], [kpis["churn_rate"] * 100], color="red")
            ax.set_title("Churn Rate (%)")
            save(fig, "churn.png", "Attrition level", 
                 1.0 if kpis["churn_rate"] > 0.05 else 0.7, "retention")

        # 4. Ticket Volume
        if c["ticket"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Tickets"], [kpis["ticket_volume"]], color="orange")
            ax.set_title("Total Support Tickets")
            save(fig, "tickets.png", "Support load", 0.6, "support")

        # 5. First Response Time (FRT)
        if c["frt"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["frt"]].hist(ax=ax, bins=10, color="purple")
            ax.set_title("First Response Time (Hours)")
            save(fig, "frt.png", "Response speed", 
                 0.9 if kpis.get("avg_first_response_time", 0) > 24 else 0.65, "support")

        # 6. Resolution Time
        if c["art"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["art"]].hist(ax=ax, bins=10, color="teal")
            ax.set_title("Resolution Time (Hours)")
            save(fig, "art.png", "Resolution efficiency", 0.7, "support")

        # 7. Sentiment Score
        if c["sentiment"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["sentiment"]].hist(ax=ax, bins=10, color="gray")
            ax.set_title("Customer Sentiment")
            save(fig, "sentiment.png", "Sentiment tone", 
                 0.9 if kpis.get("avg_sentiment", 0) < 0 else 0.6, "cx")

        # 8. FCR Rate (FIX 2: Safety Check Applied)
        if c["fcr"] and kpis.get("first_contact_resolution_rate") is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["FCR"], [kpis["first_contact_resolution_rate"] * 100], color="gold")
            ax.set_title("First Contact Resolution (%)")
            save(fig, "fcr.png", "Resolution quality", 
                 0.85 if kpis["first_contact_resolution_rate"] < 0.7 else 0.6, "support")

        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        churn = kpis.get("churn_rate", 0)
        nps = kpis.get("avg_nps", 10)
        frt = kpis.get("avg_first_response_time", 0)
        sentiment = kpis.get("avg_sentiment", 1)

        # Composite Insights
        if churn > 0.05 and frt > 24:
            insights.append({
                "level": "CRITICAL",
                "title": "Service-Driven Churn",
                "so_what": "High churn coincides with slow support response times (>24h)."
            })

        if churn > 0.05 and nps < 0:
            insights.append({
                "level": "CRITICAL",
                "title": "Loyalty Crisis",
                "so_what": "Negative NPS and rising churn indicate systemic dissatisfaction."
            })

        # Atomic Insights
        if sentiment < -0.1:
            insights.append({
                "level": "WARNING",
                "title": "Negative Sentiment",
                "so_what": "Customer feedback is predominantly negative."
            })

        if kpis.get("first_contact_resolution_rate", 1) < 0.70:
            insights.append({
                "level": "WARNING",
                "title": "Low Resolution Quality",
                "so_what": "FCR is below 70%, meaning customers have to follow up repeatedly."
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Customer Health Stable",
                "so_what": "CX metrics are healthy."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if "Service-Driven Churn" in titles:
            recs.append({"action": "Staff up support team or deploy automation immediately.", "priority": "HIGH"})

        if "Loyalty Crisis" in titles:
            recs.append({"action": "Launch emergency win-back campaign for detractors.", "priority": "HIGH"})

        # FIX 1: NPS Threshold aligned (Negative NPS = Problem)
        if kpis.get("avg_nps", 0) < 0:
            recs.append({"action": "Analyze detractor feedback for root causes.", "priority": "MEDIUM"})

        if not recs:
            recs.append({"action": "Maintain current CX initiatives.", "priority": "LOW"})

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class CustomerDomainDetector(BaseDomainDetector):
    domain_name = "customer"
    TOKENS: Set[str] = {
        "customer", "nps", "csat", "churn",
        "ticket", "support", "sentiment", "resolution"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {c.lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        if any("nps" in c for c in cols) or any("churn" in c for c in cols):
            confidence = max(confidence, 0.95)

        return DomainDetectionResult(
            domain="customer",
            confidence=confidence,
            signals={"matched_columns": hits}
        )


def register(registry):
    registry.register(
        name="customer",
        domain_cls=CustomerDomain,
        detector_cls=CustomerDomainDetector,
    )
