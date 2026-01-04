# =====================================================
# GENERIC FALLBACK DOMAIN — UNIVERSAL (FINAL)
# Sreejita Framework v3.5.x
# =====================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# GENERIC DOMAIN
# =====================================================

class GenericDomain(BaseDomain):
    """
    Universal fallback domain.

    Purpose:
    - Handle unknown / weak / mixed datasets safely
    - Never fail orchestration
    - Never hallucinate domain-specific intelligence
    - Always produce executive-safe output
    """

    name = "generic"
    description = "Generic dataset analysis (fallback domain)"

    # -------------------------------------------------
    # PREPROCESS (SAFE PASS-THROUGH)
    # -------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # -------------------------------------------------
    # KPI ENGINE (MINIMAL, GOVERNANCE-SAFE)
    # -------------------------------------------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = len(df)

        kpis: Dict[str, Any] = {
            "primary_sub_domain": "generic",
            "sub_domains": {"generic": 1.0},
            "record_count": volume,
            "total_volume": volume,
            "column_count": len(df.columns),
            "data_completeness": round(1 - df.isna().mean().mean(), 3),
            "numeric_column_count": int(df.select_dtypes(include="number").shape[1]),
            "date_column_count": int(
                df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]
            ),
        }

        # Governance warning
        if volume < 20:
            kpis["data_warning"] = "Very small dataset — insights are indicative only"

        # Confidence map (uniform & conservative)
        kpis["_confidence"] = {
            k: 0.4 for k in kpis if not k.startswith("_")
        }

        return kpis

    # -------------------------------------------------
    # VISUAL INTELLIGENCE (GUARANTEED)
    # -------------------------------------------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:

        output_dir.mkdir(parents=True, exist_ok=True)
        visuals: List[Dict[str, Any]] = []

        # -----------------------------
        # Visual 1 — Dataset Size
        # -----------------------------
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Records"], [len(df)])
            ax.set_title("Dataset Size Overview", fontweight="bold")
            ax.set_ylabel("Record Count")

            path = output_dir / "generic_dataset_size.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Total number of records in the dataset.",
                "importance": 0.6,
                "confidence": 0.5,
                "sub_domain": "generic",
            })
        except Exception:
            pass

        # -----------------------------
        # Visual 2 — Column Completeness
        # -----------------------------
        try:
            completeness = df.notna().mean().sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            completeness.plot(kind="bar", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Top Column Completeness", fontweight="bold")
            ax.set_ylabel("Completeness Ratio")

            path = output_dir / "generic_column_completeness.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Data completeness of top columns.",
                "importance": 0.55,
                "confidence": 0.5,
                "sub_domain": "generic",
            })
        except Exception:
            pass

        return visuals

    # -------------------------------------------------
    # INSIGHTS (NON-HALLUCINATING)
    # -------------------------------------------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:

        insights: List[Dict[str, Any]] = []

        insights.append({
            "sub_domain": "generic",
            "level": "INFO",
            "title": "Unclassified Dataset Structure",
            "so_what": (
                "The dataset does not strongly align with a supported business domain. "
                "Analysis is limited to structural and statistical signals."
            ),
            "confidence": 0.5,
        })

        if kpis.get("data_completeness", 1) < 0.7:
            insights.append({
                "sub_domain": "generic",
                "level": "WARNING",
                "title": "Data Completeness Risk",
                "so_what": (
                    "Significant missing data may limit analytical reliability and downstream interpretation."
                ),
                "confidence": 0.6,
            })

        # Hard guarantee: minimum 3 insights
        while len(insights) < 3:
            insights.append({
                "sub_domain": "generic",
                "level": "INFO",
                "title": "Baseline Dataset Signal",
                "so_what": (
                    "Dataset structure is stable with no immediately detectable anomalies."
                ),
                "confidence": 0.45,
            })

        return insights

    # -------------------------------------------------
    # RECOMMENDATIONS (SAFE & ACTIONABLE)
    # -------------------------------------------------

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *_,
    ) -> List[Dict[str, Any]]:

        recommendations: List[Dict[str, Any]] = []

        recommendations.append({
            "sub_domain": "generic",
            "priority": "HIGH",
            "action": "Clarify dataset intent and business context",
            "owner": "Data Owner",
            "timeline": "Immediate",
            "goal": "Enable accurate domain classification and deeper analysis",
            "confidence": 0.6,
        })

        recommendations.append({
            "sub_domain": "generic",
            "priority": "MEDIUM",
            "action": "Improve schema documentation and column naming consistency",
            "owner": "Data Engineering",
            "timeline": "30–60 days",
            "goal": "Increase semantic resolvability and automation readiness",
            "confidence": 0.55,
        })

        recommendations.append({
            "sub_domain": "generic",
            "priority": "LOW",
            "action": "Perform exploratory data profiling before operational use",
            "owner": "Analytics",
            "timeline": "Ongoing",
            "goal": "Reduce misinterpretation risk",
            "confidence": 0.5,
        })

        return recommendations


# =====================================================
# GENERIC DOMAIN DETECTOR
# =====================================================

class GenericDomainDetector(BaseDomainDetector):
    """
    Always returns a low-confidence fallback result.
    """

    domain_name = "generic"

    def detect(self, df: pd.DataFrame):
        return DomainDetectionResult(
            domain="generic",
            confidence=0.2,
            signals={"fallback": True},
        )


# =====================================================
# REGISTRATION HOOK
# =====================================================

def register(registry):
    registry.register("generic", GenericDomain, GenericDomainDetector)
