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
# CUSTOMER / CX DOMAIN (v3.0 - FULL AUTHORITY)
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
            df = _prepare_time_series(df, self.time_col)
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

        # 2. Satisfaction Metrics (Pre-Normalization)
        kpis["target_satisfaction_score"] = 4.0 # Benchmark (1-5 scale)

        if satisfaction and pd.api.types.is_numeric_dtype(df[satisfaction]):
            sat_series = df[satisfaction].copy()
            avg_raw = sat_series.mean()

            # Normalize 1-10 scale to 1-5 scale for consistency
            if avg_raw > 5:
                sat_series = sat_series / 2

            kpis["avg_satisfaction_score"] = sat_series.mean()
            
            # Thresholds apply correctly regardless of original scale
            kpis["low_satisfaction_rate"] = (sat_series < 3).mean()
            kpis["high_satisfaction_rate"] = (sat_series > 4.5).mean()

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
            if customer:
                ticket_counts = df.groupby(customer)[ticket].count()
                kpis["avg_tickets_per_customer"] = ticket_counts.mean()
                kpis["high_ticket_customer_rate"] = (ticket_counts > 3).mean()
            else:
                kpis["total_tickets"] = len(df)

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

        # -------- Visual 1: Satisfaction Distribution (Normalized) --------
        if satisfaction and pd.api.types.is_numeric_dtype(df[satisfaction]):
            p = output_dir / "satisfaction_distribution.png"

            plt.figure(figsize=(7, 4))
            
            # Normalize data for plotting to match KPI scale (1-5)
            plot_series = df[satisfaction].dropna()
            if plot_series.mean() > 5:
                plot_series = plot_series / 2
                
            plot_series.plot(kind="hist", bins=10, color="#1f77b4", edgecolor='white')
            plt.title("Customer Satisfaction Distribution (Normalized 1-5)")
            plt.xlabel("Satisfaction Score")
            plt.xlim(0, 5) # Lock axis to standard scale
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

    # ---------------- ATOMIC INSIGHTS (WITH DOMINANCE RULE) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        # === STEP 1: Composite FIRST (Authority Layer) ===
        composite: List[Dict[str, Any]] = []
        # Only run deep analysis if we have enough customers (or rows)
        base_size = kpis.get("customer_count", len(df))
        if base_size > 30:
            composite = self.generate_composite_insights(df, kpis)

        dominant_titles = {
            i["title"] for i in composite
            if i["level"] in {"RISK", "WARNING"}
        }

        # === STEP 2: Suppression Rules ===
        # If "Service-Driven Churn" is found, suppress generic "High Churn"
        suppress_churn = "Service-Driven Churn Risk" in dominant_titles
        
        # If "Silent Attrition" is found, suppress atomic "Low Satisfaction" or "Churn"
        suppress_sat = "Silent Attrition Detected" in dominant_titles
        if suppress_sat:
            suppress_churn = True # Silent attrition IS the churn story

        churn = kpis.get("churn_rate")
        sat = kpis.get("avg_satisfaction_score")
        low_sat = kpis.get("low_satisfaction_rate")

        # === STEP 3: Guarded Atomic Insights ===
        
        # Churn Insight
        if churn is not None and not suppress_churn:
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

        # Satisfaction Insight
        if sat is not None and sat < 3.5 and not suppress_sat:
            insights.append({
                "level": "WARNING",
                "title": "Low Customer Satisfaction",
                "so_what": f"Average satisfaction is {sat:.2f}/5."
            })

        if low_sat is not None and low_sat > 0.20 and not suppress_sat:
            insights.append({
                "level": "WARNING",
                "title": "Large Dissatisfied Segment",
                "so_what": f"{low_sat:.1%} of customers report low satisfaction."
            })

        # === STEP 4: Composite LAST (Authority Wins) ===
        insights += composite

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Customer Experience Stable",
                "so_what": "Satisfaction and retention metrics are within healthy ranges."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (CX v3.0) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Customer v3 Composite Intelligence Layer.
        Detects Service-Driven Churn, Price Sensitivity, etc.
        """
        insights: List[Dict[str, Any]] = []

        churn = kpis.get("churn_rate")
        high_ticket_rate = kpis.get("high_ticket_customer_rate")
        high_sat_rate = kpis.get("high_satisfaction_rate")

        # 1. Service-Driven Churn: High Support Volume + High Churn
        if high_ticket_rate is not None and churn is not None:
            if high_ticket_rate > 0.15 and churn > 0.08:
                insights.append({
                    "level": "RISK",
                    "title": "Service-Driven Churn Risk",
                    "so_what": (
                        f"15%+ of customers have high support ticket volume, and churn "
                        f"is elevated ({churn:.1%}). Support issues may be driving attrition."
                    )
                })

        # 2. Price/Value Mismatch: High Satisfaction but High Churn
        if high_sat_rate is not None and churn is not None:
            if high_sat_rate > 0.40 and churn > 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Potential Price/Value Mismatch",
                    "so_what": (
                        f"Customer satisfaction is high ({high_sat_rate:.1%} happy customers), "
                        f"yet churn is also high ({churn:.1%}). Competitors may be undercutting on price."
                    )
                })

        # 3. Silent Attrition: Low Support Volume + High Churn
        if high_ticket_rate is not None and churn is not None:
            if high_ticket_rate < 0.05 and churn > 0.10:
                insights.append({
                    "level": "RISK",
                    "title": "Silent Attrition Detected",
                    "so_what": (
                        f"Churn is high ({churn:.1%}) but support volume is low. "
                        f"Customers are leaving without complaining (Silent Churn)."
                    )
                })

        return insights

    # ---------------- RECOMMENDATIONS (AUTHORITY BASED) ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []

        # 1. Check Composite Context
        composite = []
        base_size = kpis.get("customer_count", len(df))
        if base_size > 30:
            composite = self.generate_composite_insights(df, kpis)
        
        titles = [i["title"] for i in composite]

        # AUTHORITY RULES: Mandatory Actions
        if "Service-Driven Churn Risk" in titles:
            return [{
                "action": "Improve support SLAs, reduce repeat tickets, and audit support workflows",
                "priority": "HIGH",
                "timeline": "Immediate"
            }]

        if "Silent Attrition Detected" in titles:
            return [{
                "action": "Launch proactive outreach campaign (NPS/Feedback) to at-risk silent customers",
                "priority": "HIGH",
                "timeline": "This Week"
            }]

        if "Potential Price/Value Mismatch" in titles:
            return [{
                "action": "Review pricing tiers and conduct competitive feature benchmarking",
                "priority": "MEDIUM",
                "timeline": "Next Quarter"
            }]

        # 2. Fallback to Atomic Recs
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
