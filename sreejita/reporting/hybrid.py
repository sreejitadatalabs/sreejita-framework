from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT (v3.5 – MD SOURCE OF TRUTH)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.5 Report Engine

    - Deterministic intelligence (v3.4)
    - Optional AI-assisted narrative layer (v3.5)
    - Markdown remains the single source of truth
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENGINE ENTRY POINT
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:

        config = config or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "Sreejita_Executive_Report.md"
        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)

            # v3.5 OPTIONAL AI NARRATIVE (FAIL-SAFE)
            self._write_optional_narrative(
                f,
                run_id,
                domain_results,
                config,
            )

            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # OPTIONAL NARRATIVE SECTION (v3.5)
    # -------------------------------------------------

    def _write_optional_narrative(
        self,
        f,
        run_id: str,
        domain_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
    ):
        narrative_cfg = config.get("narrative", {})
        if not narrative_cfg.get("enabled", False):
            return  # v3.4 behavior

        # Lazy imports (CRITICAL for optional AI)
        from sreejita.narrative.schema import (
            NarrativeInput,
            NarrativeInsight,
            NarrativeAction,
        )
        from sreejita.narrative.llm import LLMClient
        from sreejita.narrative.composer import generate_narrative

        domain = self._sort_domains(domain_results.keys())[0]
        result = domain_results.get(domain, {})

        insights = self._prioritize_insights(result.get("insights", []))
        recs = result.get("recommendations", [])

        if not insights:
            return

        narrative_input = NarrativeInput(
            run_id=run_id,
            domain=domain.replace("_", " ").title(),
            insights=[
                NarrativeInsight(
                    level=i.get("level"),
                    title=i.get("title"),
                    description=i.get("so_what"),
                )
                for i in insights
                if i.get("title") and i.get("so_what")
            ],
            actions=[
                NarrativeAction(
                    action=r.get("action"),
                    priority=r.get("priority", "HIGH"),
                    timeline=r.get("timeline", "Immediate"),
                )
                for r in recs[:1]
                if r.get("action")
            ],
            confidence_band=narrative_cfg.get("confidence_band", "MEDIUM"),
        )

        llm_client = LLMClient(narrative_cfg)

        try:
            narrative_text = generate_narrative(narrative_input, llm_client)
        except Exception:
            # v3.5 rule: AI failure must NEVER block report generation
            f.write(
                "\n> ⚠️ *AI narrative could not be generated for this run.*\n\n"
            )
            return

        f.write("\n## 🤖 AI-Assisted Narrative (Optional)\n\n")
        f.write(
            "> ⚠️ *This section is AI-assisted and optional. It summarizes existing decisions "
            "using a language model. No new metrics or recommendations are introduced. *\n\n"
        )
        f.write(narrative_text.strip() + "\n\n")

    # -------------------------------------------------
    # HEADER & EXEC SUMMARY
    # -------------------------------------------------

    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write("# 📊 Executive Decision Report\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Run ID:** {run_id}\n")
        f.write(f"- **Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n")
        f.write("- **Framework Version:** Sreejita v3.5\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")

        f.write(
            "\n> This report presents decision-grade insights generated using "
            "**Sreejita Composite Intelligence**, focusing on risks, "
            "opportunities, and recommended actions.\n\n"
        )

    # -------------------------------------------------
    # DOMAIN SECTIONS
    # -------------------------------------------------

    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        if not isinstance(result, dict):
            f.write("_Invalid domain output._\n\n")
            return

        kpis = result.get("kpis", {})
        insights = self._prioritize_insights(result.get("insights", []))
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        f.write("### 🧠 Strategic Intelligence\n")
        if insights:
            for ins in insights:
                if not ins.get("title") or not ins.get("so_what"):
                    continue
                f.write(
                    f"#### {self._level_icon(ins.get('level'))} {ins['title']}\n"
                )
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("_Operations within normal parameters._\n\n")

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

        if isinstance(visuals, list) and visuals:
            f.write("### 👁️ Visual Evidence\n")
            for idx, vis in enumerate(visuals[:2], start=1):
                path = vis.get("path")
                if not path:
                    continue
                img = f"visuals/{Path(path).name}"
                caption = vis.get("caption", "Visualization")
                f.write(f"![{caption}]({img})\n")
                f.write(f"> *Fig {idx}.1 — {caption}*\n\n")

        if isinstance(recs, list) and recs:
            primary = recs[0]
            if "action" in primary:
                f.write("### 🚀 Action Plan\n")
                f.write("| Action | Priority | Timeline |\n")
                f.write("| :--- | :--- | :--- |\n")
                f.write(
                    f"| {primary['action']} | "
                    f"{primary.get('priority', 'HIGH')} | "
                    f"{primary.get('timeline', 'Immediate')} |\n\n"
                )

    # -------------------------------------------------
    # FOOTER & HELPERS
    # -------------------------------------------------

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write(
            "_Prepared by **Sreejita Data Labs** · "
            "Framework v3.5 · Confidential_\n"
        )

    def _prioritize_insights(self, insights: List[Dict[str, Any]]):
        order = {"RISK": 0, "WARNING": 1, "INFO": 2}
        return sorted(insights, key=lambda i: order.get(i.get("level"), 3))[:5]

    def _sort_domains(self, domains):
        priority = ["finance", "retail", "ecommerce", "supply_chain", "healthcare"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _level_icon(self, level: str):
        return {"RISK": "🔴", "WARNING": "🟠", "INFO": "🔵"}.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any):
        if isinstance(v, (int, float)):
            abs_v = abs(v)

            # Percentages
            if any(
                x in key.lower()
                for x in ["rate", "ratio", "margin", "conversion"]
            ) and abs_v <= 2:
                return f"{v:.1%}"

            # Millions
            if abs_v >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"

            # Thousands
            if abs_v >= 1_000:
                return f"{v / 1_000:.1f}K"

            # Small numbers
            if isinstance(v, float):
                return f"{v:.2f}"

            return f"{v:,}"

        return str(v)



# =====================================================
# BACKWARD-COMPATIBLE ENTRY POINT (v3.5.1)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    v3.5.1 Stable entry:
    - Generates Markdown
    - Builds Executive PDF payload
    """

    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    domain_results = generate_report_payload(input_path, config)

    engine = HybridReport()
    md_path = Path(
        engine.build(domain_results, run_dir, config.get("metadata"), config)
    )

    # -------------------------------
    # BUILD PDF PAYLOAD (CRITICAL)
    # -------------------------------
    primary_domain = engine._sort_domains(domain_results.keys())[0]
    result = domain_results.get(primary_domain, {})

    payload = {
        "meta": {
            "domain": primary_domain.replace("_", " ").title(),
        },
        "summary": [
            ins.get("title")
            for ins in result.get("insights", [])[:5]
            if ins.get("title")
        ],
        "kpis": result.get("kpis", {}),
        "visuals": result.get("visuals", []),
        "insights": result.get("insights", []),
        "recommendations": result.get("recommendations", []),
    }

    return {
        "markdown": str(md_path),
        "payload": payload,
        "run_dir": str(run_dir),
    }
