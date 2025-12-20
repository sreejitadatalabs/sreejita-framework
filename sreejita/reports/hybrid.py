from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT — v3.2 (FINAL, STABLE)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Executive Report Engine (v3.2)

    Responsibilities:
    - Convert domain intelligence into executive narrative
    - Decision-first structure (Insights → KPIs → Visuals → Actions)
    - Zero data loading, zero domain routing
    """

    name = "hybrid"

    # -------------------------------------------------
    # FRAMEWORK CONTRACT
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:

        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be pathlib.Path")

        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"Sreejita_Executive_Report_{datetime.now():%Y-%m-%d}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, metadata)

            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {})
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # SECTIONS
    # -------------------------------------------------

    def _write_header(self, f, metadata: Optional[Dict[str, Any]]):
        f.write("# 📊 Executive Decision Report\n")
        f.write(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}\n\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write(
            "> ⚠️ **Executive Summary**: This report uses **v3 Composite Intelligence** "
            "to prioritize root-cause risks over raw metrics.\n\n"
        )

    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        kpis = result.get("kpis", {})
        insights = result.get("insights", [])
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # INSIGHTS
        insights = self._prioritize_insights(insights)
        if insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for i in insights:
                if "title" in i and "so_what" in i:
                    f.write(f"#### {self._level_icon(i.get('level'))} {i['title']}\n")
                    f.write(f"{i['so_what']}\n\n")
        else:
            f.write("✅ _No critical risks detected._\n\n")

        # KPIs
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n|---|---|\n")
            for k, v in list(kpis.items())[:8]:
                f.write(f"| {k.replace('_',' ').title()} | **{self._format_value(k, v)}** |\n")
            f.write("\n")

        # VISUALS
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for v in visuals[:2]:
                img = Path(v.get("path")).name
                caption = v.get("caption", "Visualization")
                f.write(f"![{caption}]({img})\n> *{caption}*\n\n")

        # ACTIONS
        if recs:
            f.write("### 🚀 Required Actions\n")
            primary = recs[0]
            f.write(f"**PRIMARY MANDATE:** {primary.get('action')}\n\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Powered by Sreejita Framework v3.2_")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------

    def _prioritize_insights(self, insights: List[Dict[str, Any]]):
        order = {"RISK": 0, "WARNING": 1, "INFO": 2}
        return sorted(insights, key=lambda i: order.get(i.get("level"), 99))[:5]

    def _sort_domains(self, domains):
        priority = [
            "finance", "retail", "ecommerce", "supply_chain",
            "marketing", "hr", "customer", "healthcare"
        ]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _level_icon(self, level: str):
        return {"RISK": "🔴", "WARNING": "🟠", "INFO": "🔵"}.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any):
        if isinstance(v, float):
            if any(x in key.lower() for x in ["rate", "ratio", "margin", "percent"]) and abs(v) <= 2:
                return f"{v:.1%}"
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)


# =====================================================
# SINGLE ENTRY POINT (CLI / UI / BATCH)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Path:
    from sreejita.reporting.orchestrator import generate_report_payload

    domain_results = generate_report_payload(
        input_path=input_path,
        config=config
    )

    output_root = Path(config.get("output_dir", "runs"))
    run_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    return HybridReport().build(
        domain_results=domain_results,
        output_dir=run_dir,
        metadata=config.get("metadata")
    )
