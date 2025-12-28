from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport
from sreejita.narrative.engine import build_narrative


# =====================================================
# HYBRID REPORT ENGINE (v3.6.1 — EXECUTIVE-GRADE)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Report Engine

    - Markdown = Source of Truth
    - JSON Payload = PDF / UI / API
    - Deterministic Intelligence (No LLM hallucinations)
    - Executive-first narrative
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENGINE ENTRY POINT (MARKDOWN ONLY)
    # -------------------------------------------------
    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        narrative_data: Any,          # Narrative object (single source of truth)
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "Sreejita_Executive_Report.md"
        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)

            # ---------------- EXECUTIVE NARRATIVE ----------------
            self._write_narrative(f, narrative_data)

            # ---------------- DOMAIN SECTIONS ----------------
            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # EXECUTIVE NARRATIVE SECTION
    # -------------------------------------------------
    def _write_narrative(self, f, narrative):
        f.write("\n## 🧭 Executive Narrative\n\n")

        # Executive Summary (MANDATORY)
        if hasattr(narrative, "executive_summary") and narrative.executive_summary:
            for line in narrative.executive_summary:
                f.write(f"- {line}\n")
            f.write("\n")

        # Financial Impact
        if hasattr(narrative, "financial_impact") and narrative.financial_impact:
            f.write("### 💰 Financial Impact\n")
            for line in narrative.financial_impact:
                f.write(f"- {line}\n")
            f.write("\n")

        # Risks
        if hasattr(narrative, "risks") and narrative.risks:
            f.write("### ⚠️ Key Risks\n")
            for r in narrative.risks:
                f.write(f"- {r}\n")
            f.write("\n")

    # -------------------------------------------------
    # DOMAIN SECTION
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

        # 1. INSIGHTS (BRAIN)
        f.write("### 🧠 Strategic Insights\n")
        if insights:
            for ins in insights:
                icon = self._level_icon(ins.get("level"))
                title = ins.get("title", "Insight")
                f.write(f"#### {icon} {title}\n")
                f.write(f"{ins.get('so_what')}\n\n")
        else:
            f.write("_No material risks detected._\n\n")

        # 2. VISUALS (EVIDENCE)
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for idx, vis in enumerate(visuals[:4], start=1):
                path = vis.get("path")
                if not path:
                    continue
                caption = vis.get("caption", "Visualization")
                img_name = Path(path).name
                f.write(f"![{caption}](visuals/{img_name})\n")
                f.write(f"> *Fig {idx} — {caption}*\n\n")

        # 3. KPIs (DATA)
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in list(kpis.items())[:12]:
                f.write(f"| {k.replace('_',' ').title()} | **{self._format_value(k, v)}** |\n")
            f.write("\n")

        # 4. RECOMMENDATIONS (ACTION)
        if recs:
            primary = recs[0]
            f.write("### 🚀 Recommendation Snapshot\n")
            f.write(f"- **Action:** {primary.get('action')}\n")
            f.write(f"- **Priority:** {primary.get('priority', 'HIGH')}\n")
            f.write(f"- **Timeline:** {primary.get('timeline', 'Immediate')}\n\n")

    # -------------------------------------------------
    # HEADER & FOOTER
    # -------------------------------------------------
    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write("# 📊 Executive Decision Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Run ID:** {run_id}\n")
        f.write(f"- **Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_',' ').title()}**: {v}\n")

        f.write("\n> Decision-grade insights generated using **Sreejita Composite Intelligence**.\n\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Prepared by **Sreejita Data Labs** · Framework v3.6.1 · Confidential_\n")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def _prioritize_insights(self, insights: List[Dict[str, Any]]):
        order = {"CRITICAL": 0, "RISK": 1, "WARNING": 2, "INFO": 3}
        return sorted(insights, key=lambda i: order.get(i.get("level"), 4))[:5]

    def _sort_domains(self, domains):
        priority = ["finance", "retail", "supply_chain", "ecommerce", "healthcare", "marketing"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _level_icon(self, level: str):
        return {"CRITICAL": "🔥", "RISK": "🔴", "WARNING": "🟠", "INFO": "🔵"}.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any):
        if isinstance(v, (int, float)):
            abs_v = abs(v)

            if any(x in key.lower() for x in ["rate", "ratio", "margin", "ctr", "roas"]):
                if abs_v <= 5:
                    return f"{v:.1%}" if "ratio" not in key else f"{v:.2f}x"

            if abs_v >= 1_000_000:
                return f"{v/1_000_000:.1f}M"
            if abs_v >= 1_000:
                return f"{v/1_000:.1f}K"

            return f"{v:,.0f}"

        return str(v)


# =====================================================
# STABLE ENTRY POINT (PAYLOAD + MARKDOWN)
# =====================================================
def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Domain Outputs
    domain_results = generate_report_payload(input_path, config)

    # 2. Identify Primary Domain
    engine = HybridReport()
    primary_domain = engine._sort_domains(domain_results.keys())[0]
    result = domain_results.get(primary_domain, {})

    # 3. Build Narrative (SINGLE SOURCE OF TRUTH)
    narrative = build_narrative(
        domain=primary_domain,
        kpis=result.get("kpis", {}),
        insights=result.get("insights", []),
        recommendations=result.get("recommendations", []),
    )

    # 4. Generate Markdown Report
    md_path = engine.build(
        domain_results,
        narrative,
        run_dir,
        config.get("metadata"),
        config,
    )

    # 5. PDF / UI Payload (Step C — CORRECT LOCATION)
    payload = {
        "meta": {
            "domain": primary_domain.replace("_", " ").title(),
            "run_id": f"RUN-{datetime.utcnow():%H%M%S}",
        },
        "summary": narrative.executive_summary,
        "narrative": narrative,
        "kpis": result.get("kpis", {}),
        "visuals": result.get("visuals", []),
        "insights": narrative.key_findings,
        "recommendations": narrative.action_plan,
        "risks": narrative.risks,
        "financial_impact": narrative.financial_impact,
    }

    return {
        "markdown": str(md_path),
        "payload": payload,
        "run_dir": str(run_dir),
    }
