# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL, AUTHORITATIVE)
# Sreejita Framework v3.5.x
# =====================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sreejita.narrative.executive_cognition import (
    build_executive_payload,
    build_subdomain_executive_payloads,
)

# =====================================================
# BASE DOMAIN
# =====================================================

class BaseDomain(ABC):
    """
    Universal BaseDomain contract.

    Responsibilities:
    - KPI computation
    - Insight generation
    - Recommendation generation
    - Visual intelligence
    - Executive cognition assembly

    MUST NOT:
    - Perform routing
    - Perform orchestration
    - Import sub-domain engines
    """

    name: str = "generic"
    description: str = "Generic domain"

    def __init__(self):
        self._last_kpis: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # OPTIONAL VALIDATION (SAFE DEFAULT)
    # --------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Optional domain-level validation.
        Default = always valid.
        """
        return True

    # --------------------------------------------------
    # OPTIONAL PREPROCESS
    # --------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional preprocessing hook.
        Default = passthrough.
        """
        return df

    # --------------------------------------------------
    # REQUIRED DOMAIN CONTRACTS
    # --------------------------------------------------
    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute KPIs.
        MUST return a dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from KPIs.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Generate visual intelligence.
        """
        raise NotImplementedError

    # --------------------------------------------------
    # 🔒 UNIVERSAL VISUAL SAFETY NET (CRITICAL)
    # --------------------------------------------------
    def ensure_minimum_visuals(
        self,
        visuals: List[Dict[str, Any]],
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Guarantees at least 2 visuals exist.
        Absolute last-resort fallback.
        NEVER raises.
        """

        visuals = visuals if isinstance(visuals, list) else []

        if len(visuals) >= 2:
            return visuals

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Fallback 1 — Dataset Size
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Records"], [len(df)])
            ax.set_title("Dataset Size Overview", fontweight="bold")
            ax.set_ylabel("Record Count")

            path = output_dir / f"{self.name}_fallback_records.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Total number of records (fallback evidence).",
                "importance": 0.3,
                "confidence": 0.4,
                "sub_domain": self.name,
            })

            # Fallback 2 — Data Completeness
            completeness = 1 - df.isna().mean().mean()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Completeness"], [completeness])
            ax.set_ylim(0, 1)
            ax.set_title("Data Completeness Indicator", fontweight="bold")

            path = output_dir / f"{self.name}_fallback_completeness.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Overall data completeness ratio (fallback evidence).",
                "importance": 0.3,
                "confidence": 0.4,
                "sub_domain": self.name,
            })

        except Exception:
            # Absolute safety: visuals must never crash execution
            pass

        return visuals

    # --------------------------------------------------
    # 🧠 EXECUTIVE COGNITION (GLOBAL + SUB-DOMAIN)
    # --------------------------------------------------
    def build_executive(
        self,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Universal executive cognition builder.

        Guarantees:
        - Global executive payload
        - Per-sub-domain executive payloads (if present)
        """

        executive = build_executive_payload(
            kpis=kpis,
            insights=insights or [],
            recommendations=recommendations or [],
        )

        sub_domains = kpis.get("sub_domains")

        if isinstance(sub_domains, dict) and sub_domains:
            executive["executive_by_sub_domain"] = (
                build_subdomain_executive_payloads(
                    kpis=kpis,
                    insights=insights or [],
                    recommendations=recommendations or [],
                )
            )
        else:
            executive["executive_by_sub_domain"] = {}

        return executive

    # --------------------------------------------------
    # LEGACY PIPELINE (OPTIONAL, SAFE)
    # --------------------------------------------------
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optional legacy pipeline.
        Orchestrator may bypass this.
        """

        if not self.validate_data(df):
            raise ValueError(f"Data validation failed for domain: {self.name}")

        df = self.preprocess(df)
        kpis = self.calculate_kpis(df)
        insights = self.generate_insights(df, kpis)
        recommendations = self.generate_recommendations(df, kpis, insights)

        executive = self.build_executive(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
        )

        return {
            "domain": self.name,
            "description": self.description,
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "executive": executive,
        }
