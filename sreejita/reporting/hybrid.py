from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from dataclasses import asdict

from sreejita.reporting.base import BaseReport
from sreejita.narrative.engine import build_narrative


# =====================================================
# HYBRID REPORT ENGINE (v3.7 — FINAL)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Report Engine

    Responsibilities:
    - Render Markdown report
    - Package executive-ready payload
    - NEVER compute intelligence
    """

    name = "hybrid"

    # -------------------------------------------------
    # BUILD MARKDOWN REPORT
    # -------------------------------------------------
    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        narrative_data: Any,
        executive_payload: Dict[str, Any],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "Sreejita_Executive_Report.md"
        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)
            self._write_executive_snapshot(f, executive_payload)
            self._write_narrative(f, narrative_data)

            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # EXECUTIVE SNAPSHOT (PAGE 1 CONTENT)
    # -------------------------------------------------
    def _write_executive_snapshot(self, f, payload: Dict[str, Any]):
        snap = payload.get("decision_snapshot")
        if not snap:
            return

        f.write("## Executive Decision Snapshot\n\n")

        f.write(f"**Overall Risk:** {snap.get('overall_risk', '-')}\n\n")

        if snap.get("top_problems"):
            f.write("### Top Problems\n")
            for p in snap["top_problems"]:
                f.write(f"- {p}\n")

        if snap.get("top_actions"):
            f.write("\n### Top Actions (90 Days)\n")
            for a in snap["top_actions"]:
                f.write(f"- {a}\n")

        if snap.get("decisions_required"):
            f.write("\n### Decisions Required\n")
            for d in snap["decisions_required"]:
                f.write(f"- [ ] {d}\n")

        f.write(
            "\n**Confidence Scale:** "
            "85–100 = 🟢 Green | 70–84 = 🟡 Yellow | "
            "50–69 = 🟠 Orange | <50 = 🔴 Red\n"
        )

        f.write("\n---\n\n")

    # -------------------------------------------------
    # EXECUTIVE NARRATIVE
    # -------------------------------------------------
    def _write_narrative(self, f, narrative):
        f.write("## Executive Brief\n\n")

        for line in narrative.executive_summary or []:
            f.write(f"- {line}\n")

        if narrative.financial_impact:
            f.write("\n### Financial Impact\n")
            for line in narrative.financial_impact:
                f.write(f"- {line}\n")

        if narrative.risks:
            f.write("\n### Strategic Risks\n")
            for r in narrative.risks:
                f.write(f"- {r}\n")

        if narrative.action_plan:
            f.write("\n### Action Plan\n")
            f.write("| Action | Owner | Timeline | Success Metric |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")

            for a in narrative.action_plan:
                row = a if isinstance(a, dict) else asdict(a)
                f.write(
                    f"| {row.get('action')} | {row.get('owner')} | "
                    f"{row.get('timeline')} | {row.get('success_kpi')} |\n"
                )

        f.write("\n---\n\n")

    # -------------------------------------------------
    # DOMAIN SECTION
    # -------------------------------------------------
    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write(f"## Domain Deep Dive — {domain.replace('_',' ').title()}\n\n")

        kpis = result.get("kpis", {}) or {}
        visuals = result.get("visuals", []) or []

        if kpis:
            f.write("### Key Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in list(kpis.items())[:12]:
                f.write(f"| {k.replace('_',' ').title()} | {self._format_value(k, v)} |\n")
            f.write("\n")

        if visuals:
            f.write("### Visual Evidence\n")
            for vis in visuals[:6]:
                f.write(f"![{vis.get('caption')}]({vis.get('path')})\n")
                f.write(f"> {vis.get('caption')}\n\n")

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
        f.write("_Generated by **Sreejita Intelligence Engine** · Framework v3.7_\n")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def _sort_domains(self, domains):
        priority = ["healthcare", "finance", "sales", "marketing"]
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
    Intelligence is already computed by orchestrator.
    """
    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    domain_results = generate_report_payload(input_path, config)

    engine = HybridReport()
    primary_domain = engine._sort_domains(domain_results.keys())[0]
    primary = domain_results.get(primary_domain, {})

    narrative = build_narrative(
        domain=primary_domain,
        kpis=primary.get("kpis", {}),
        insights=primary.get("insights", []),
        recommendations=primary.get("recommendations", []),
    )

    executive_payload = primary.get("executive", {})

    md_path = engine.build(
        domain_results=domain_results,
        narrative_data=narrative,
        executive_payload=executive_payload,
        output_dir=run_dir,
        metadata=config.get("metadata"),
    )

    return {
        "markdown": str(md_path),
        "payload": {
            "executive_snapshot": executive_payload.get("decision_snapshot"),
            "primary_kpis": executive_payload.get("executive_kpis"),
            "summary": narrative.executive_summary,
            "visuals": primary.get("visuals", []),
            "insights": narrative.key_findings,
            "recommendations": primary.get("recommendations", []),
            "scorecard": {
                "risk_label": executive_payload
                .get("decision_snapshot", {})
                .get("overall_risk", "-")
            },
        },
        "run_dir": str(run_dir),
    }
