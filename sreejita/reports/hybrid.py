import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT (v3.3 - MARKDOWN + PDF)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.3 Report Engine

    - Decision-first narrative
    - Composite intelligence
    - Robust formatting
    - Markdown as source of truth
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENTRY POINT (ENGINE)
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:

        # SAFETY: ensure Path
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

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
            "to prioritize root-cause risks over raw metrics.\n\n"
        )

    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        kpis = result.get("kpis", {})
        insights = result.get("insights", [])
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # 1. Strategic Intelligence
        insights = self._prioritize_insights(insights)

        if insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for ins in insights:
                if not ins.get("title") or not ins.get("so_what"):
                    continue
                icon = self._level_icon(ins.get("level"))
                f.write(f"#### {icon} {ins['title']}\n")
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("✅ _Operations within normal parameters._\n\n")

        # 2. KPIs
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            clean_kpis = {k: v for k, v in kpis.items() if not k.startswith("_")}

            for k, v in list(clean_kpis.items())[:8]:
                label = k.replace("_", " ").title()
                f.write(f"| {label} | **{self._format_value(k, v)}** |\n")

            f.write("\n")

        # 3. Visuals
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for vis in visuals[:2]:
                img = Path(vis["path"]).name
                caption = vis.get("caption", "Visualization")
                f.write(f"![{caption}]({img})\n")
                f.write(f"> *{caption}*\n\n")

        # 4. Actions
        if recs:
            f.write("### 🚀 Required Actions\n")

            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            recs = sorted(
                recs,
                key=lambda r: priority_order.get(r.get("priority", "LOW"), 3)
            )

            primary = recs[0]
            f.write(f"**PRIMARY MANDATE:** {primary['action']}\n")
            f.write(f"- Priority: {primary.get('priority', 'HIGH')}\n")
            f.write(f"- Timeline: {primary.get('timeline', 'Immediate')}\n\n")

            for r in recs[1:3]:
                f.write(f"- {r['action']} ({r.get('timeline', 'Ongoing')})\n")

            f.write("\n")

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
# PDF RENDERER
# =====================================================

def render_pdf(md_path: Path) -> Optional[Path]:
    pdf_path = md_path.with_suffix(".pdf")

    try:
        subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=xelatex"],
            check=True
        )
        return pdf_path
    except Exception:
        return None


# =====================================================
# BACKWARD-COMPATIBLE ENTRY POINT (CLI / UI / BATCH)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Path:
    """
    This is what CLI, BatchRunner, UI call.
    """

    from sreejita.reporting.orchestrator import generate_report_payload

    # 1. Run analysis
    domain_results = generate_report_payload(
        input_path=input_path,
        config=config
    )

    # 2. Output dir
    output_root = Path(config.get("output_dir", "runs"))
    run_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Build markdown
    engine = HybridReport()
    md_path = engine.build(domain_results, run_dir, config.get("metadata"))

    # 4. Build PDF (optional)
    pdf_path = render_pdf(md_path)

    return pdf_path if pdf_path else md_path
