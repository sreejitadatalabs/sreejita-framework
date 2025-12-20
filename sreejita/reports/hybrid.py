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
    - Markdown as source of truth
    - PDF as optional renderer
    """

    name = "hybrid"

    # -------------------------------------------------
    # CORE ENGINE
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"Sreejita_Executive_Report_{datetime.now():%Y-%m-%d}.md"

        with report_path.open("w", encoding="utf-8") as f:
            self._write_header(f, metadata)

            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(f, domain, domain_results.get(domain, {}))

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
            "> ⚠️ **Executive Summary**: Powered by v3 Composite Intelligence. "
            "Root-cause risks are prioritized over raw metrics.\n\n"
        )

    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        kpis = result.get("kpis", {})
        insights = self._prioritize_insights(result.get("insights", []))
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        if insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for i in insights:
                if not i.get("title") or not i.get("so_what"):
                    continue
                f.write(f"#### {self._level_icon(i.get('level'))} {i['title']}\n")
                f.write(f"{i['so_what']}\n\n")
        else:
            f.write("✅ _Operations within normal parameters._\n\n")

        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n| :--- | :--- |\n")
            for k, v in list(kpis.items())[:8]:
                f.write(f"| {k.replace('_',' ').title()} | **{self._format_value(k, v)}** |\n")
            f.write("\n")

        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for v in visuals[:2]:
                img = Path(v["path"]).name
                f.write(f"![{v.get('caption','Chart')}]({img})\n\n")

        if recs:
            f.write("### 🚀 Required Actions\n")
            recs = sorted(recs, key=lambda r: {"HIGH":0,"MEDIUM":1,"LOW":2}.get(r.get("priority","LOW"),3))
            primary = recs[0]
            f.write(f"**PRIMARY MANDATE:** {primary['action']}\n")
            f.write(f"- Priority: {primary.get('priority','HIGH')}\n")
            f.write(f"- Timeline: {primary.get('timeline','Immediate')}\n\n")

    def _write_footer(self, f):
        f.write("\n---\n_Powered by Sreejita Framework v3.3_\n")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------

    def _prioritize_insights(self, insights):
        order = {"RISK":0,"WARNING":1,"INFO":2}
        return sorted(insights, key=lambda i: order.get(i.get("level"),3))[:5]

    def _sort_domains(self, domains):
        priority = ["finance","retail","ecommerce","supply_chain"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _level_icon(self, level):
        return {"RISK":"🔴","WARNING":"🟠","INFO":"🔵"}.get(level,"ℹ️")

    def _format_value(self, key, v):
        if isinstance(v, float):
            if any(x in key.lower() for x in ["rate","ratio","margin","conversion"]) and abs(v) <= 2:
                return f"{v:.1%}"
            return f"{v:,.2f}"
        return f"{v:,}" if isinstance(v,int) else str(v)


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
# ENTRY POINT (CLI / UI / BATCH)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Path:
    from sreejita.reporting.orchestrator import generate_report_payload

    domain_results = generate_report_payload(input_path, config)

    run_dir = Path(config.get("output_dir","runs")) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    engine = HybridReport()
    md = engine.build(domain_results, run_dir, config.get("metadata"))

    pdf = render_pdf(md)
    return pdf if pdf else md
