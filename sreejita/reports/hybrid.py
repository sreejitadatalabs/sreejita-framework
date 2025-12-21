from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT (v3.3 – MARKDOWN SOURCE OF TRUTH)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.3 Report Engine

    - Decision-first narrative
    - Composite intelligence
    - Markdown is the single source of truth
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENGINE ENTRY POINT
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_name = f"Sreejita_Executive_Report_{datetime.now():%Y-%m-%d}.md"
        report_path = output_dir / report_name

        print(f"📝 Writing Markdown report to: {report_path}")

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, metadata)

            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {})
                )

            self._write_footer(f)

        print("✅ Markdown report generated successfully")
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

        if not isinstance(result, dict):
            f.write("⚠️ _Invalid domain output._\n\n")
            return

        kpis = result.get("kpis", {})
        insights = self._prioritize_insights(result.get("insights", []))
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # Strategic Intelligence
        if insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for ins in insights:
                if not ins.get("title") or not ins.get("so_what"):
                    continue
                f.write(
                    f"#### {self._level_icon(ins.get('level'))} {ins['title']}\n"
                )
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("✅ _Operations within normal parameters._\n\n")

        # KPIs
        if isinstance(kpis, dict) and kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            for k, v in list(kpis.items())[:8]:
                f.write(
                    f"| {k.replace('_', ' ').title()} | "
                    f"**{self._format_value(k, v)}** |\n"
                )
            f.write("\n")

        # Visuals
        if isinstance(visuals, list) and visuals:
            f.write("### 👁️ Visual Evidence\n")
            for vis in visuals[:2]:
                path = vis.get("path")
                if not path:
                    continue
                img = Path(path).name
                caption = vis.get("caption", "Visualization")
                f.write(f"![{caption}]({img})\n")
                f.write(f"> *{caption}*\n\n")

        # Actions
        if isinstance(recs, list) and recs:
            primary = recs[0]
            if "action" in primary:
                f.write("### 🚀 Required Actions\n")
                f.write(f"**PRIMARY MANDATE:** {primary['action']}\n")
                f.write(f"- Priority: {primary.get('priority', 'HIGH')}\n")
                f.write(f"- Timeline: {primary.get('timeline', 'Immediate')}\n\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Powered by Sreejita Framework v3.3_")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------

    def _prioritize_insights(self, insights: List[Dict[str, Any]]):
        order = {"RISK": 0, "WARNING": 1, "INFO": 2}
        return sorted(insights, key=lambda i: order.get(i.get("level"), 3))[:5]

    def _sort_domains(self, domains):
        priority = ["finance", "retail", "ecommerce", "supply_chain"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _level_icon(self, level: str):
        return {"RISK": "🔴", "WARNING": "🟠", "INFO": "🔵"}.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any):
        if isinstance(v, float):
            if any(x in key.lower() for x in ["rate", "ratio", "margin", "conversion"]) and abs(v) <= 2:
                return f"{v:.1%}"
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)


# =====================================================
# BACKWARD-COMPATIBLE ENTRY POINT (NO PDF HERE)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Path:
    """
    Single entry point used by CLI, BatchRunner, Streamlit UI.
    Generates Markdown ONLY.
    """

    print("🔥 Hybrid run() called")

    from sreejita.reporting.orchestrator import generate_report_payload

    domain_results = generate_report_payload(
        input_path=input_path,
        config=config
    )

    output_root = Path(config.get("output_dir", "runs"))
    run_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    engine = HybridReport()
    md_path = engine.build(domain_results, run_dir, config.get("metadata"))

    print("📄 MD PATH:", md_path)
    return md_path
