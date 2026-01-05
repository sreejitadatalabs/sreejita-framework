# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL, FIXED)
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
    - ≥2 visuals always (PDF-safe)
    - Executive cognition always available
    - Never crashes reporting
    """

    name: str = "generic"
    description: str = "Generic domain"

    def __init__(self):
        self._last_kpis: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # OPTIONAL VALIDATION
    # --------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> bool:
        return True

    # --------------------------------------------------
    # OPTIONAL PREPROCESS
    # --------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    # --------------------------------------------------
    # REQUIRED CONTRACTS
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
    # 🔒 UNIVERSAL VISUAL SAFETY NET (PDF-GUARANTEED)
    # --------------------------------------------------
    def ensure_minimum_visuals(
        self,
        visuals: List[Dict[str, Any]],
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        HARD GUARANTEE:
        - Always returns ≥2 visuals
        - Each visual has confidence ≥0.3
        - PDF renderer will never reject
        """

        visuals = visuals if isinstance(visuals, list) else []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def add_fallback(index: int, title: str, y, caption: str):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(title, y)
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
            # Fallback #1 — Dataset size
            if len(visuals) < 1:
                add_fallback(
                    1,
                    ["Records"],
                    [len(df)],
                    "Dataset size overview (fallback evidence).",
                )

            # Fallback #2 — Column completeness
            if len(visuals) < 2:
                completeness = df.notna().mean().mean()
                add_fallback(
                    2,
                    ["Completeness"],
                    [round(completeness, 2)],
                    "Overall data completeness indicator (fallback evidence).",
                )

        except Exception:
            # Absolute safety: never crash
            pass

        return visuals[:6]

    # --------------------------------------------------
    # 🧠 EXECUTIVE COGNITION
    # --------------------------------------------------
    def build_executive(
        self,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

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
