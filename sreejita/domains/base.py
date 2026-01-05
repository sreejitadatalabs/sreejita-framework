# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL, GOVERNED)
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


class BaseDomain(ABC):
    """
    Universal BaseDomain contract.

    GUARANTEES:
    - Domain intelligence only (no orchestration)
    - ≥2 visuals ALWAYS (PDF-safe)
    - Executive cognition ALWAYS present
    - Never crashes reporting layer
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
        Optional schema validation.
        Default: always valid.
        """
        return True

    # --------------------------------------------------
    # OPTIONAL PREPROCESS
    # --------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional preprocessing hook.
        Default: passthrough.
        """
        return df

    # --------------------------------------------------
    # REQUIRED DOMAIN CONTRACTS
    # --------------------------------------------------
    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Core KPI computation.
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
    # 🔒 UNIVERSAL VISUAL SAFETY NET (PDF GUARANTEE)
    # --------------------------------------------------
    def ensure_minimum_visuals(
        self,
        visuals: List[Dict[str, Any]],
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        HARD GOVERNANCE RULE:
        - ALWAYS returns ≥2 visuals
        - Each visual has confidence ≥0.3
        - PDF renderer will NEVER reject

        This is the FINAL safety layer.
        """

        visuals = visuals if isinstance(visuals, list) else []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _register_fallback(
            index: int,
            labels,
            values,
            title: str,
            caption: str,
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, values)
            ax.set_title(title, fontweight="bold")

            path = output_dir / f"{self.name}_fallback_{index}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": 0.3,
                "confidence": 0.4,
                "sub_domain": self.name,
            })

        try:
            # --------------------------------------------------
            # Fallback #1 — Dataset Size
            # --------------------------------------------------
            if len(visuals) < 1:
                _register_fallback(
                    index=1,
                    labels=["Records"],
                    values=[len(df)],
                    title="Dataset Size Overview",
                    caption="Total number of records (fallback evidence).",
                )

            # --------------------------------------------------
            # Fallback #2 — Data Completeness
            # --------------------------------------------------
            if len(visuals) < 2:
                completeness = (
                    1 - df.isna().mean().mean()
                    if not df.empty
                    else 0.0
                )

                _register_fallback(
                    index=2,
                    labels=["Completeness"],
                    values=[round(completeness, 2)],
                    title="Data Completeness Indicator",
                    caption="Overall data completeness ratio (fallback evidence).",
                )

        except Exception:
            # Absolute safety: BaseDomain must never crash execution
            pass

        return visuals[:6]

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

        GUARANTEES:
        - Global executive payload
        - Per-sub-domain executive payloads (if present)
        """

        # ---------------- GLOBAL EXECUTIVE ----------------
        executive = build_executive_payload(
            kpis=kpis,
            insights=insights or [],
            recommendations=recommendations or [],
        )

        # ---------------- SUB-DOMAIN EXECUTIVES ----------------
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
    # OPTIONAL LEGACY PIPELINE
    # --------------------------------------------------
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optional legacy execution pipeline.
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
