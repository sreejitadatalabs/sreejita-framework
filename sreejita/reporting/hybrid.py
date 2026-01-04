from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT ENGINE — UNIVERSAL (FINAL, HARDENED)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Report Engine (Authoritative)

    Responsibilities:
    - Render executive-ready Markdown
    - Enforce narrative ordering
    - Render global + per-sub-domain executive cognition
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
            raise RuntimeError(
                f"HybridReport.build expected dict, got {type(domain_results)}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"
        report_path = output_dir / f"Sreejita_Executive_Report_{run_id}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)

            domains = list(domain_results.keys())
            if not domains:
                raise RuntimeError("No domains returned by orchestrator")

            primary_domain = self._sort_domains(domains)[0]
            primary = domain_results.get(primary_domain, {}) or {}

            # =================================================
            # GLOBAL EXECUTIVE SUMMARY (PRIMARY DOMAIN)
            # =================================================
            executive = primary.get("executive", {})
            if isinstance(executive, dict):
                self._write_global_executive(f, executive)

            # =================================================
            # PER-SUB-DOMAIN EXECUTIVE SECTIONS (CRITICAL)
            # =================================================
            self._write_sub_domain_executives(f, executive)

            # =================================================
            # DOMAIN DEEP DIVES
            # =================================================
            for domain in self._sort_domains(domains):
                payload = domain_results.get(domain)
                if isinstance(payload, dict):
                    self._write_domain_section(f, domain, payload)

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # GLOBAL EXECUTIVE
    # -------------------------------------------------
    def _write_global_executive(self, f, executive: Dict[str, Any]):
        brief = executive.get("executive_brief")
        board = executive.get("board_readiness", {})

        if isinstance(brief, str) and brief.strip():
            f.write("## Executive Summary\n\n")
            f.write(f"{brief}\n\n")

        if isinstance(board, dict) and board:
            f.write("### Board Readiness\n")
            f.write(f"- **Score:** {board.get('score', '-')} / 100\n")
            f.write(f"- **Status:** {board.get('band', '-')}\n\n")

        trend = executive.get("board_readiness_trend")
        if isinstance(trend, dict):
            f.write(
                f"- **Trend:** {trend.get('trend','→')} "
                f"(Prev: {trend.get('previous_score','—')}, "
                f"Current: {trend.get('current_score','—')})\n\n"
            )

        f.write("---\n\n")

    # -------------------------------------------------
    # SUB-DOMAIN EXECUTIVE SECTIONS (AUTHORITATIVE)
    # -------------------------------------------------
    def _write_sub_domain_executives(self, f, executive: Dict[str, Any]):
        exec_by_sub = executive.get("executive_by_sub_domain")

        if not isinstance(exec_by_sub, dict) or not exec_by_sub:
            return

        f.write("## Executive Summary by Operating Area\n\n")

        for sub, payload in exec_by_sub.items():
            if not isinstance(payload, dict):
                continue

            brief = payload.get("executive_brief")
            board = payload.get("board_readiness", {})

            f.write(f"### {sub.replace('_',' ').title()}\n\n")

            if isinstance(brief, str) and brief.strip():
                f.write(f"{brief}\n\n")

            if isinstance(board, dict):
                f.write(
                    f"- **Board Readiness Score:** {board.get('score','-')} / 100  \n"
                    f"- **Status:** {board.get('band','-')}\n\n"
                )

        f.write("---\n\n")

    # -------------------------------------------------
    # DOMAIN SECTION
    # -------------------------------------------------
    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write(f"## Domain Deep Dive — {domain.replace('_',' ').title()}\n\n")

        # ---------------- KPIs ----------------
        raw_kpis = result.get("kpis")
        kpis = (
            {
                k: v for k, v in raw_kpis.items()
                if isinstance(k, str) and not k.startswith("_")
            }
            if isinstance(raw_kpis, dict)
            else {}
        )

        if kpis:
            f.write("### Key Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            shown = 0
            for k, v in kpis.items():
                if shown >= 9:
                    break
                f.write(
                    f"| {k.replace('_',' ').title()} | {self._format_value(k, v)} |\n"
                )
                shown += 1

            while shown < 5:
                f.write("| Data Coverage | Insufficient signal |\n")
                shown += 1

            f.write("\n")

        # ---------------- VISUALS ----------------
        visuals = result.get("visuals", [])
        if isinstance(visuals, list) and visuals:
            f.write("### Visual Evidence\n")
            for vis in visuals[:6]:
                if not isinstance(vis, dict):
                    continue
                path = vis.get("path")
                if not path:
                    continue
                caption = vis.get("caption", "Visual evidence")
                confidence = int(float(vis.get("confidence", 0)) * 100)

                f.write(f"![{caption}]({path})\n")
                f.write(f"> {caption} (Confidence: {confidence}%)\n\n")

        # ---------------- INSIGHTS ----------------
        insights = result.get("insights", [])
        if isinstance(insights, list) and insights:
            f.write("### Key Insights\n")
            for ins in insights[:5]:
                if not isinstance(ins, dict):
                    continue
                f.write(
                    f"- **{ins.get('level','INFO')}** — "
                    f"{ins.get('title','')}: {ins.get('so_what','')}\n"
                )
            f.write("\n")

        # ---------------- RECOMMENDATIONS ----------------
        recs = result.get("recommendations", [])
        if isinstance(recs, list) and recs:
            f.write("### Recommendations\n")
            for r in recs[:5]:
                if not isinstance(r, dict):
                    continue
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

        if isinstance(metadata, dict):
            for k, v in metadata.items():
                f.write(f"- **{str(k).replace('_',' ').title()}**: {v}\n")

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
# PUBLIC ENTRY POINT (BACKWARD COMPATIBILITY)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stable public API for:
    - CLI
    - Batch runner
    - Scheduler
    - File watcher
    - Tests
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

    domains = list(domain_results.keys())
    primary_domain = engine._sort_domains(domains)[0]
    primary = domain_results.get(primary_domain, {}) or {}

    return {
        "markdown": str(md_path),
        "domain_results": domain_results,
        "primary_domain": primary_domain,
        "executive": primary.get("executive", {}),
        "visuals": primary.get("visuals", []),
        "insights": primary.get("insights", []),
        "recommendations": primary.get("recommendations", []),
        "run_dir": str(run_dir),
    }
