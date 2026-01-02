from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT ENGINE — UNIVERSAL (FINAL)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Report Engine (Authoritative)

    Responsibilities:
    - Render executive-ready Markdown
    - Enforce narrative ordering
    - NEVER compute intelligence
    """

    name = "hybrid"

    # -------------------------------------------------
    # BUILD MARKDOWN REPORT
    # -------------------------------------------------
    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"
        report_path = output_dir / f"Sreejita_Executive_Report_{run_id}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)

            # Primary domain drives executive cognition
            primary_domain = self._sort_domains(domain_results.keys())[0]
            primary = domain_results.get(primary_domain, {})
            executive = primary.get("executive", {})

            self._write_executive_brief(f, executive)
            self._write_board_readiness(f, executive)

            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # EXECUTIVE BRIEF (PAGE 1)
    # -------------------------------------------------
    def _write_executive_brief(self, f, executive: Dict[str, Any]):
        brief = executive.get("executive_brief")
        if not brief:
            return

        f.write("## Executive Brief\n\n")
        f.write(f"{brief}\n\n")
        f.write("---\n\n")

    # -------------------------------------------------
    # BOARD READINESS
    # -------------------------------------------------
    def _write_board_readiness(self, f, executive: Dict[str, Any]):
        br = executive.get("board_readiness")
        if not br:
            return

        f.write("## Board Readiness Assessment\n\n")
        f.write(f"- **Score:** {br.get('score', '-')} / 100\n")
        f.write(f"- **Status:** {br.get('band', '-')}\n")
        f.write("\n---\n\n")

    # -------------------------------------------------
    # DOMAIN SECTION
    # -------------------------------------------------
    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write(f"## Domain Deep Dive — {domain.replace('_',' ').title()}\n\n")

        kpis = {
            k: v for k, v in (result.get("kpis") or {}).items()
            if not k.startswith("_")
        }
        visuals = result.get("visuals", []) or []
        insights = result.get("insights", []) or []
        recs = result.get("recommendations", []) or []

        # ---------------- KPIs ----------------
        if kpis:
            f.write("### Key Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            for k, v in list(kpis.items())[:9]:
                f.write(
                    f"| {k.replace('_',' ').title()} | "
                    f"{self._format_value(k, v)} |\n"
                )
            f.write("\n")

        # ---------------- VISUAL EVIDENCE ----------------
        if visuals:
            f.write("### Visual Evidence\n")
            for vis in visuals[:6]:
                f.write(f"![{vis.get('caption')}]({vis.get('path')})\n")
                conf = int(vis.get("confidence", 0) * 100)
                f.write(
                    f"> {vis.get('caption')} "
                    f"(Confidence: {conf}%)\n\n"
                )

        # ---------------- INSIGHTS ----------------
        if insights:
            f.write("### Key Insights\n")
            for ins in insights[:5]:
                f.write(
                    f"- **{ins.get('level','INFO')}** — "
                    f"{ins.get('title')}: {ins.get('so_what')}\n"
                )
            f.write("\n")

        # ---------------- RECOMMENDATIONS ----------------
        if recs:
            f.write("### Recommendations\n")
            for r in recs[:5]:
                f.write(
                    f"- **{r.get('priority')}** — {r.get('action')} "
                    f"(Owner: {r.get('owner')}, "
                    f"Timeline: {r.get('timeline')})\n"
                )
            f.write("\n")

        f.write("---\n\n")

    # -------------------------------------------------
    # HEADER & FOOTER
    # -------------------------------------------------
    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write("# Sreejita Executive Report\n\n")
        f.write(
            f"**Run ID:** `{run_id}` | "
            f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n\n"
        )

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_',' ').title()}**: {v}\n")

        f.write("\n---\n\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Generated by **Sreejita Universal Domain Intelligence**_\n")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def _sort_domains(self, domains):
        priority = ["healthcare", "finance", "retail", "marketing"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _format_value(self, key: str, v: Any):
        if isinstance(v, (int, float)):
            if "rate" in key:
                return f"{v:.1%}"
            if abs(v) >= 1_000_000:
                return f"{v/1_000_000:.1f}M"
            if abs(v) >= 1_000:
                return f"{v/1_000:.1f}K"
            return f"{v:.2f}"
        return str(v)


# =====================================================
# PUBLIC ENTRY POINT
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thin glue layer.
    Intelligence is computed by orchestrator.
    """
    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    domain_results = generate_report_payload(input_path, config)

    engine = HybridReport()
    md_path = engine.build(
        domain_results=domain_results,
        output_dir=run_dir,
        metadata=config.get("metadata"),
    )

    primary_domain = engine._sort_domains(domain_results.keys())[0]
    primary = domain_results.get(primary_domain, {})

    return {
        "markdown": str(md_path),
        "payload": {
            "executive": primary.get("executive", {}),
            "visuals": primary.get("visuals", []),
            "insights": primary.get("insights", []),
            "recommendations": primary.get("recommendations", []),
        },
        "run_dir": str(run_dir),
    }
