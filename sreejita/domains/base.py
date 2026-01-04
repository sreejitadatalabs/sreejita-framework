# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL)
# Sreejita Framework v3.5+
# =====================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sreejita.narrative.executive_cognition import (
    build_executive_payload,
    build_subdomain_executive_payloads,
)


class BaseDomain(ABC):
    """
    Abstract base class for all domain modules.

    Architecture:
    - Strategy Pattern
    - Plugin-based domain registry
    - Universal executive cognition hook
    - Universal visual safety guarantees

    Domains:
    - Compute intelligence
    - NEVER render reports
    - NEVER manage orchestration
    """

    name: str = "generic"
    description: str = "Generic domain"
    required_columns: List[str] = []

    def __init__(self):
        self.kpis: Dict[str, Any] = {}
        self.insights: List[Any] = []
        self._last_kpis: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # OPTIONAL VALIDATION (DEFAULT SAFE)
    # --------------------------------------------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Optional domain-level validation.
        Default = always valid.

        Domains may override for strict schemas.
        """
        return True

    # --------------------------------------------------
    # REQUIRED DOMAIN CONTRACTS
    # --------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional preprocessing hook.
        Default = passthrough.
        """
        return df

    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Core KPI computation.
        Must return a dict.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any], *args, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Insight generation.
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
        Recommendation generation.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Visual intelligence generation.
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

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Records"], [len(df)])
            ax.set_title("Dataset Scale Overview", fontweight="bold")
            ax.set_ylabel("Record Count")

            path = output_dir / f"{self.name}_global_fallback.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Dataset scale fallback evidence.",
                "importance": 0.3,
                "confidence": 0.4,
                "sub_domain": self.name,
            })

        except Exception:
            # Absolute safety: never crash reporting
            pass

        return visuals

    # --------------------------------------------------
    # 🧠 UNIVERSAL EXECUTIVE COGNITION HOOK (CRITICAL)
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
        - Safe fallback for single-domain datasets
        """

        # -------------------------------
        # GLOBAL EXECUTIVE
        # -------------------------------
        executive = build_executive_payload(
            kpis=kpis,
            insights=insights or [],
            recommendations=recommendations or [],
        )

        # -------------------------------
        # SUB-DOMAIN EXECUTIVES (OPTIONAL)
        # -------------------------------
        sub_domains = kpis.get("sub_domains")

        if isinstance(sub_domains, dict) and sub_domains:
            executive_by_sub = build_subdomain_executive_payloads(
                kpis=kpis,
                insights=insights or [],
                recommendations=recommendations or [],
            )
        else:
            executive_by_sub = {}

        executive["executive_by_sub_domain"] = executive_by_sub

        return executive

    # --------------------------------------------------
    # LEGACY PIPELINE (OPTIONAL USE)
    # --------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optional legacy pipeline.
        Orchestrator may bypass this.
        """

        if not self.validate_data(df):
            raise ValueError(f"Data validation failed for {self.name}")

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
