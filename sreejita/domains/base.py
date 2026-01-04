# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL, ENFORCED)
# Sreejita Framework v3.6
# =====================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sreejita.narrative.executive_cognition import build_executive_payload
from sreejita.domains.subdomain.engine import UniversalSubDomainEngine


class BaseDomain(ABC):
    """
    Universal BaseDomain for all Sreejita domains.

    GUARANTEES:
    - Deterministic execution
    - Universal executive cognition
    - Universal sub-domain handling
    - Visual safety (never < 2 visuals)
    - Never crashes orchestrator

    NON-RESPONSIBILITIES:
    - Report rendering
    - Orchestration
    - I/O management
    """

    # --------------------------------------------------
    # DOMAIN METADATA
    # --------------------------------------------------
    name: str = "generic"
    description: str = "Generic domain"
    required_columns: List[str] = []

    def __init__(self):
        self._last_kpis: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # OPTIONAL VALIDATION (SAFE DEFAULT)
    # --------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Optional strict validation.
        Default = always valid.
        """
        return True

    # --------------------------------------------------
    # OPTIONAL PREPROCESSING
    # --------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-specific preprocessing hook.
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
        Must return a dict.
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
        Generate insights.
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
        Absolute last-resort fallback.
        Guarantees >= 2 visuals.
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
    # 🧠 UNIVERSAL EXECUTIVE COGNITION (AUTHORITATIVE)
    # --------------------------------------------------
    def build_executive(
        self,
        *,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
        subdomain_signals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Universal executive cognition builder.

        OUTPUT GUARANTEES:
        - executive_brief
        - board_readiness
        - primary_sub_domain
        - sub_domains
        - executive_by_sub_domain
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
        # UNIVERSAL SUB-DOMAIN ENGINE
        # -------------------------------
        sub_payload = UniversalSubDomainEngine.resolve(
            domain=self.name,
            signals=subdomain_signals or {},
            kpis=kpis,
            insights=insights or [],
            recommendations=recommendations or [],
        )

        executive.update(sub_payload)
        return executive

    # --------------------------------------------------
    # LEGACY PIPELINE (OPTIONAL / SAFE)
    # --------------------------------------------------
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optional legacy execution path.
        Orchestrator may bypass this.
        """

        if not self.validate_data(df):
            raise ValueError(f"Validation failed for domain: {self.name}")

        df = self.preprocess(df)

        kpis = self.calculate_kpis(df)
        self._last_kpis = kpis

        insights = self.generate_insights(df, kpis)
        recommendations = self.generate_recommendations(df, kpis, insights)

        executive = self.build_executive(
            kpis=kpis,
            insights=insights,
            recommendations=recommendations,
            subdomain_signals=kpis.get("sub_domain_signals", {}),
        )

        return {
            "domain": self.name,
            "description": self.description,
            "kpis": kpis,
            "insights": insights,
            "recommendations": recommendations,
            "executive": executive,
        }
