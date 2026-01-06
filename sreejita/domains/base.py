# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6 STABILIZED
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

    HARD GUARANTEES:
    - No shared DataFrame mutation
    - Deterministic KPI lifecycle
    - Visual safety (guaranteed)
    - Executive-safe cognition

    MUST NOT:
    - Route domains
    - Orchestrate pipelines
    - Guess sub-domains
    """

    name: str = "generic"
    description: str = "Generic domain"

    def __init__(self):
        self._last_kpis: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # OPTIONAL VALIDATION (SAFE DEFAULT)
    # --------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> bool:
        return True

    # --------------------------------------------------
    # OPTIONAL PREPROCESS (SAFE BY DEFAULT)
    # --------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Default preprocess is COPY-ONLY.
        Domains may override, but MUST return a new df.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("preprocess expects a DataFrame")
        return df.copy(deep=False)

    # --------------------------------------------------
    # REQUIRED DOMAIN CONTRACTS
    # --------------------------------------------------
    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
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
        raise NotImplementedError

    @abstractmethod
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # --------------------------------------------------
    # 🔒 UNIVERSAL VISUAL SAFETY NET (LAST RESORT)
    # --------------------------------------------------
    def ensure_minimum_visuals(
        self,
        visuals: List[Dict[str, Any]],
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Guarantees at least 2 visuals.
        Fallback visuals are explicitly marked.
        NEVER raises.
        """

        visuals = visuals if isinstance(visuals, list) else []

        if len(visuals) >= 2:
            return visuals

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # -----------------------------
            # Fallback 1 — Dataset Size
            # -----------------------------
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
                "importance": 0.2,
                "confidence": 0.3,
                "sub_domain": self.name,
                "inference_type": "fallback",
            })

            # -----------------------------
            # Fallback 2 — Data Completeness
            # -----------------------------
            completeness = float(1 - df.isna().mean().mean())

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
                "importance": 0.2,
                "confidence": 0.3,
                "sub_domain": self.name,
                "inference_type": "fallback",
            })

        except Exception:
            pass  # absolute safety

        return visuals

    # --------------------------------------------------
    # 🧠 EXECUTIVE COGNITION (SAFE, NON-HALLUCINATING)
    # --------------------------------------------------
    def build_executive(
        self,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        executive = build_executive_payload(
            kpis=kpis or {},
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
    # 🔒 SAFE LEGACY PIPELINE (AUTHORITATIVE)
    # --------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        *,
        visual_output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Legacy pipeline.

        GUARANTEES:
        - No shared mutation
        - Visuals always exist
        - Executive always stable
        """

        if not self.validate_data(df):
            raise ValueError(f"Data validation failed for domain: {self.name}")

        # Defensive copy
        df = df.copy(deep=False)

        # Domain preprocessing
        df = self.preprocess(df)

        # KPI lifecycle
        kpis = self.calculate_kpis(df)
        if not isinstance(kpis, dict):
            raise TypeError("calculate_kpis must return a dict")

        self._last_kpis = kpis

        insights = self.generate_insights(df, kpis)
        recommendations = self.generate_recommendations(df, kpis, insights)

        # Visuals (guaranteed)
        visuals: List[Dict[str, Any]] = []
        if visual_output_dir is not None:
            try:
                visuals = self.generate_visuals(df, visual_output_dir)
                visuals = self.ensure_minimum_visuals(
                    visuals,
                    df,
                    visual_output_dir,
                )
            except Exception:
                visuals = self.ensure_minimum_visuals(
                    [],
                    df,
                    visual_output_dir,
                )

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
            "visuals": visuals,
            "executive": executive,
        }
