import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain


class CapabilityDrivenDomain(BaseDomain):
    """
    v2.x generic domain engine.
    Domains define CAPABILITIES, not hard logic.
    """

    CAPABILITIES = {}      # overridden per domain
    KPI_RULES = []         # overridden per domain
    VISUAL_RULES = []      # overridden per domain
    INSIGHT_RULES = []     # overridden per domain

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.capabilities = {
            name: resolve_column(df, col) is not None
            for name, col in self.CAPABILITIES.items()
        }
        return df

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis = {}
        for rule in self.KPI_RULES:
            if rule["when"](self.capabilities):
                val = rule["compute"](df)
                if val is not None:
                    kpis[rule["name"]] = val
        return kpis

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict]:
        visuals = []
        for rule in self.VISUAL_RULES:
            if rule["when"](self.capabilities):
                v = rule["plot"](df, output_dir)
                if v:
                    visuals.append(v)
        return visuals

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict]:
        insights = []
        for rule in self.INSIGHT_RULES:
            i = rule(df, kpis, self.capabilities)
            if i:
                insights.append(i)
        return insights
