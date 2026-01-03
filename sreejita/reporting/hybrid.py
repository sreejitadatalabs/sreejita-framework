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
    - Adapt orchestrator output to reporting layers
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

        if not isinstance(domain_results, dict):
            raise RuntimeError("HybridReport.build expected domain_results dict")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"
        report_path = output_dir / f"Sreejita_Executive_Report_{run_id}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)

            # ---------------- PRIMARY DOMAIN ----------------
            primary_domain = self._sort_domains(domain_results.keys())[0]
            primary = domain_results.get(primary_domain)

            # HARD SAFETY — recover instead of crash
            if not isinstance(primary, dict):
                # Attempt unwrap if orchestrator returned nested payload
                if isinstance(domain_results, dict) and len(domain_results) == 1:
                    candidate = next(iter(domain_results.values()))
                    if isinstance(candidate, dict):
                        primary = candidate
                    else:
                        primary = {}
                else:
                    primary = {}


            executive = primary.get("executive", {}) or {}

            self._write_executive_brief(f, executive)
            self._write_board_readiness(f, executive)

            # ---------------- DOMAIN SECTIONS ----------------
            for domain in self._sort_domains(domain_results.keys()):
                payload = domain_results.get(domain, {})
                if isinstance(payload, dict):
                    self._write_domain_section(f, domain, payload)

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
        if not isinstance(br, dict):
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
                path = vis.get("path")
                if not path:
                    continue

                f.write(f"![{vis.get('caption','Visual')}]({path})\n")
                conf = int(vis.get("confidence", 0) * 100)
                f.write(
                    f"> {vis.get('caption','')} "
                    f"(Confidence: {conf}%)\n\n"
                )

        # ---------------- INSIGHTS ----------------
        # ---------------- INSIGHTS ----------------
        if insights:
            f.write("### Key Insights\n")
        
            # Executive cognition returns structured insight blocks
            if isinstance(insights, dict):
                ordered = []
                ordered.extend(insights.get("strengths", []))
                ordered.extend(insights.get("warnings", []))
                ordered.extend(insights.get("risks", []))
            elif isinstance(insights, list):
                ordered = insights
            else:
                ordered = []
        
            for ins in ordered[:5]:
                f.write(
                    f"- **{ins.get('level','INFO')}** — "
                    f"{ins.get('title','')}: {ins.get('so_what','')}\n"
                )
        
            f.write("\n")

        # ---------------- RECOMMENDATIONS ----------------
        if recs:
            f.write("### Recommendations\n")
            for r in recs[:5]:
                f.write(
                    f"- **{r.get('priority','')}** — {r.get('action','')} "
                    f"(Owner: {r.get('owner','-')}, "
                    f"Timeline: {r.get('timeline','-')})\n"
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
        return sorted(
            domains,
            key=lambda d: priority.index(d) if d in priority else 99
        )

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
# PUBLIC ENTRY POINT (USED BY CLI / UI)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thin glue layer.
    Intelligence is computed by orchestrator.
    """

    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # 🔒 AUTHORITATIVE DOMAIN RESULTS
    domain_results = generate_report_payload(input_path, config)

    engine = HybridReport()
    md_path = engine.build(
        domain_results=domain_results,
        output_dir=run_dir,
        metadata=config.get("metadata"),
    )

    # 🔒 SAFE PRIMARY DOMAIN EXTRACTION
    primary_domain = engine._sort_domains(domain_results.keys())[0]
    primary = domain_results.get(primary_domain, {}) or {}

    return {
        "markdown": str(md_path),
        "domain_results": domain_results,   # KEEP FULL STRUCTURE
        "primary_domain": primary_domain,
        "executive": primary.get("executive", {}),
        "visuals": primary.get("visuals", []),
        "insights": primary.get("insights", []),
        "recommendations": primary.get("recommendations", []),
        "run_dir": str(run_dir),
    }
