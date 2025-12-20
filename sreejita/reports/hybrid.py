from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sreejita.reporting.base import BaseReport

# =====================================================
# HYBRID REPORT (v3.2 – ROBUST DECISION ENGINE)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.2 Report Engine

    Responsibilities:
    - Convert domain intelligence into executive narrative
    - Present insights BEFORE metrics (decision-first)
    - Keep visuals and actions tightly scoped
    - Never load data or infer domains (handled by Orchestrator)
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENTRY POINT (FRAMEWORK CONTRACT)
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Build the Hybrid executive report.
        """

        # Safety guard
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"Sreejita_Executive_Report_{datetime.now():%Y-%m-%d}.md"
        report_path = output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, metadata)

            ordered_domains = self._sort_domains(domain_results.keys())

            for domain in ordered_domains:
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
            "to prioritize root-cause risks over raw metrics. "
            "Actions listed below are authoritative mandates.\n\n"
        )

    def _write_domain_section(
        self,
        f,
        domain: str,
        result: Dict[str, Any]
    ):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        kpis = result.get("kpis", {})
        insights = result.get("insights", [])
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # 1️⃣ Strategic Intelligence (WHY)
        authoritative_insights = self._prioritize_insights(insights)

        if authoritative_insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for ins in authoritative_insights:
                if "title" not in ins or "so_what" not in ins:
                    continue
                icon = self._level_icon(ins.get("level"))
                f.write(f"#### {icon} {ins['title']}\n")
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("✅ _Operations within normal parameters._\n\n")

        # 2️⃣ KPIs (WHAT)
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            clean_kpis = {
                k: v for k, v in kpis.items()
                if not k.startswith("_")
            }

            for k, v in list(clean_kpis.items())[:8]:
                label = k.replace("_", " ").title().replace("Kpi", "KPI")
                value = self._format_value(k, v)
                f.write(f"| {label} | **{value}** |\n")

            f.write("\n")

        # 3️⃣ Visual Evidence (PROOF)
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for vis in visuals[:2]:
                img_name = Path(vis.get("path")).name
                caption = vis.get("caption", "Data Visualization")
                f.write(f"![{caption}]({img_name})\n")
                f.write(f"> *{caption}*\n\n")

        # 4️⃣ Required Actions (HOW)
        if recs:
            f.write("### 🚀 Required Actions\n")

            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            sorted_recs = sorted(
                recs,
                key=lambda r: priority_order.get(r.get("priority", "LOW"), 3)
            )

            primary = sorted_recs[0]
            f.write(f"**PRIMARY MANDATE:** {primary['action']}\n")
            f.write(f"- 🚨 **Priority:** {primary.get('priority', 'HIGH')}\n")
            f.write(f"- ⏱️ **Timeline:** {primary.get('timeline', 'Immediate')}\n\n")

            if len(sorted_recs) > 1:
                f.write("**Supporting Actions:**\n")
                for r in sorted_recs[1:3]:
                    f.write(f"- {r['action']} ({r.get('timeline', 'Ongoing')})\n")
                f.write("\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Powered by Sreejita Framework v3.2 | Enterprise Decision Engine_")

    # -------------------------------------------------
    # INTELLIGENCE LOGIC
    # -------------------------------------------------

    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not insights:
            return []

        def rank(i):
            return {"RISK": 0, "WARNING": 1, "INFO": 2}.get(i.get("level"), 3)

        return sorted(insights, key=rank)[:5]

    def _sort_domains(self, domains):
        priority = [
            "finance",
            "retail",
            "ecommerce",
            "supply_chain",
            "marketing",
            "hr",
            "customer",
            "healthcare"
        ]

        def sorter(d):
            return priority.index(d) if d in priority else 99

        return sorted(domains, key=sorter)

    def _level_icon(self, level: str) -> str:
        return {
            "RISK": "🔴",
            "WARNING": "🟠",
            "INFO": "🔵"
        }.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any) -> str:
        if isinstance(v, float):
            key_l = key.lower()
            if any(x in key_l for x in [
                "rate", "ratio", "margin", "percent",
                "pct", "share", "bounce", "conversion"
            ]) and abs(v) <= 2.0:
                return f"{v:.1%}"
            return f"{v:,.2f}"

        if isinstance(v, int):
            return f"{v:,}"

        return str(v)


# =====================================================
# BACKWARD-COMPATIBLE ENTRY POINT (CLI / UI / BATCH)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Path:
    """
    v3-compatible entry point.
    This is the ONLY place Hybrid touches input_path.
    """

    from sreejita.reporting.orchestrator import generate_report_payload

    domain_results = generate_report_payload(
        input_path=input_path,
        config=config
    )

    output_root = Path(config.get("output_dir", "runs"))
    run_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    engine = HybridReport()
    return engine.build(
        domain_results=domain_results,
        output_dir=run_dir,
        metadata=config.get("metadata")
    )
