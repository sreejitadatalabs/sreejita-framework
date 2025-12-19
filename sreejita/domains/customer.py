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
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    CX-safe time detector.
    Used for customer trend / churn trend only if present.
    """
    candidates = [
        "signup date", "created date", "interaction date",
        "date", "timestamp"
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
# CUSTOMER / CX DOMAIN
# =====================================================

class CustomerDomain(BaseDomain):
    name = "customer"
    description = "Customer Experience & Retention Analytics (Satisfaction, Churn, Support)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        CX data requires customer signals, not revenue.
        """
        return any(
            resolve_column(df, c) is not None
            for c in [
                "customer", "customer_id",
                "satisfaction", "rating", "score",
                "churn", "status",
                "ticket", "complaint", "support"
            ]
        )

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

        customer = resolve_column(df, "customer") or resolve_column(df, "customer_id")
        satisfaction = (
            resolve_column(df, "satisfaction")
            or resolve_column(df, "rating")
            or resolve_column(df, "score")
        )
        churn = resolve_column(df, "churn")
        status = resolve_column(df, "status")
        ticket = resolve_column(df, "ticket") or resolve_column(df, "complaint")

        # 1. Customer Base
        if customer:
            kpis["customer_count"] = df[customer].nunique()

        # 2. Satisfaction Metrics
        if satisfaction and pd.api.types.is_numeric_dtype(df[satisfaction]):
            avg_score = df[satisfaction].mean()

            # Normalize if stored as 1–10 instead of 1–5
            if avg_score > 5:
                avg_score = avg_score / 2

            kpis["avg_satisfaction_score"] = avg_score
            kpis["low_satisfaction_rate"] = (df[satisfaction] < 3).mean()

        # 3. Churn / Retention
        kpis["target_churn_rate"] = 0.05

        if churn and pd.api.types.is_numeric_dtype(df[churn]):
            kpis["churn_rate"] = df[churn].mean()

        elif status:
            # Fallback: infer churn from status text
            status_series = df[status].astype(str).str.lower()
            churned = status_series.str.contains("churn|inactive|cancel", na=False)
            kpis["churn_rate"] = churned.mean()

        # 4. Support Load
        if ticket:
            ticket_counts = df.groupby(customer)[ticket].count() if customer else df[ticket].count()
            kpis["avg_tickets_per_customer"] = ticket_counts.mean()

        return kpis

    # ---------------- VISUALS (MAX 4) ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        kpis = self.calculate_kpis(df)

        satisfaction = (
            resolve_column(df, "satisfaction")
            or resolve_column(df, "rating")
            or resolve_column(df, "score")
        )
        status = resolve_column(df, "status")
        customer = resolve_column(df, "customer") or resolve_column(df, "customer_id")
        ticket = resolve_column(df, "ticket") or resolve_column(df, "complaint")

        # -------- Visual 1: Satisfaction Distribution --------
        if satisfaction and pd.api.types.is_numeric_dtype(df[satisfaction]):
            p = output_dir / "satisfaction_distribution.png"

            plt.figure(figsize=(7, 4))
            df[satisfaction].dropna().plot(kind="hist", bins=10, color="#1f77b4")
            plt.title("Customer Satisfaction Distribution")
            plt.xlabel("Satisfaction Score")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Distribution of customer satisfaction scores"
            })

        # -------- Visual 2: Customer Status / Churn --------
        if status:
            p = output_dir / "customer_status.png"

            counts = df[status].value_counts().head(5)

            plt.figure(figsize=(7, 4))
            counts.plot(kind="bar", color="#ff7f0e")
            plt.title("Customer Status Breakdown")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Active vs inactive customer distribution"
            })

        # -------- Visual 3: Customer Trend --------
        if self.has_time_series and customer:
            p = output_dir / "customer_trend.png"

            plot_df = df.copy()
            if len(df) > 100:
                plot_df = (
                    df.set_index(self.time_col)
                    .resample("ME")
                    .nunique()
                    .reset_index()
                )

            plt.figure(figsize=(7, 4))
            plt.plot(plot_df[self.time_col], plot_df[customer], linewidth=2)
            plt.title("Customer Growth Trend")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Customer base trend over time"
            })

        # -------- Visual 4: Support Load --------
        if ticket and customer:
            p = output_dir / "support_load.png"

            top_customers = (
                df.groupby(customer)[ticket]
                .count()
                .sort_values(ascending=False)
                .head(7)
            )

            plt.figure(figsize=(7, 4))
            top_customers.plot(kind="barh", color="#d62728")
            plt.title("Customers with Highest Support Load")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Support pressure concentration"
            })

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        churn = kpis.get("churn_rate")
        sat = kpis.get("avg_satisfaction_score")
        low_sat = kpis.get("low_satisfaction_rate")

        if churn is not None:
            if churn > 0.10:
                insights.append({
                    "level": "RISK",
                    "title": "High Customer Churn",
                    "so_what": f"Churn rate is {churn:.1%}, well above the 5% target."
                })
            elif churn > 0.05:
                insights.append({
                    "level": "WARNING",
                    "title": "Rising Customer Attrition",
                    "so_what": f"Churn rate is {churn:.1%}, exceeding the acceptable threshold."
                })

        if sat is not None and sat < 3.5:
            insights.append({
                "level": "WARNING",
                "title": "Low Customer Satisfaction",
                "so_what": f"Average satisfaction is {sat:.2f}/5."
            })

        if low_sat is not None and low_sat > 0.20:
            insights.append({
                "level": "WARNING",
                "title": "Large Dissatisfied Segment",
                "so_what": f"{low_sat:.1%} of customers report low satisfaction."
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Customer Experience Stable",
                "so_what": "Satisfaction and retention metrics are within healthy ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        churn = kpis.get("churn_rate")
        low_sat = kpis.get("low_satisfaction_rate")

        if churn is not None and churn > 0.10:
            recs.append({
                "action": "Launch customer retention and win-back initiatives",
                "priority": "HIGH",
                "timeline": "Immediate"
            })

        if low_sat is not None and low_sat > 0.20:
            recs.append({
                "action": "Investigate root causes of low satisfaction and support issues",
                "priority": "MEDIUM",
                "timeline": "This Quarter"
            })

        if not recs:
            recs.append({
                "action": "Continue monitoring customer experience metrics",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class CustomerDomainDetector(BaseDomainDetector):
    domain_name = "customer"

    CUSTOMER_TOKENS: Set[str] = {
        "customer", "client", "user",
        "satisfaction", "rating", "score",
        "churn", "status",
        "ticket", "complaint", "support"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        hits = [c for c in cols if any(t in c for t in self.CUSTOMER_TOKENS)]
        confidence = min(len(hits) / 3, 1.0)

        return DomainDetectionResult(
            domain="customer",
            confidence=confidence,
            signals={"matched_columns": hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="customer",
        domain_cls=CustomerDomain,
        detector_cls=CustomerDomainDetector,
    )
