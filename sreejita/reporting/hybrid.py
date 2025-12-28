from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport
from sreejita.narrative.engine import generate_narrative


# =====================================================
# HYBRID REPORT (v3.5.1 — FINAL, DETERMINISTIC)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.5.1 Report Engine

    - Deterministic intelligence
    - Deterministic Narrative Engine (NO LLM)
    - Markdown = single source of truth
    - Narrative embedded into PDF payload
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

            # ✅ DETERMINISTIC NARRATIVE (ALWAYS RUNS)
            self._write_optional_narrative(
                f,
                run_id,
                domain_results,
                config,
            )

            # DOMAIN SECTIONS
            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # DETERMINISTIC NARRATIVE (REPLACED METHOD)
    # -------------------------------------------------
    def _write_optional_narrative(
        self,
        f,
        run_id: str,
        domain_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
    ):
        """
        Deterministic executive narrative (v3.5.1)
        LLM narrative can be layered later.
        """

        domain = self._sort_domains(domain_results.keys())[0]
        result = domain_results.get(domain, {})

        # -----------------------------
        # 1️⃣ Deterministic Narrative
        # -----------------------------
        narrative = generate_narrative(result, config)

        f.write("\n## 🧭 Executive Narrative\n\n")

        f.write("**Executive Summary**  \n")
        f.write(f"{narrative['executive_summary']}\n\n")

        f.write("**Operational Impact**  \n")
        f.write(f"{narrative['operational_impact']}\n\n")

        f.write("**Financial Impact**  \n")
        f.write(f"{narrative['financial_impact']}\n\n")

        f.write("**Risk Assessment**  \n")
        f.write(
            f"**Risk Level:** {narrative['risk_level']}  \n"
            f"{narrative['risk_statement']}\n\n"
        )

        # -----------------------------
        # 2️⃣ OPTIONAL LLM (LATER)
        # -----------------------------
        # Intentionally skipped

    # -------------------------------------------------
    # HEADER
    # -------------------------------------------------
    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write("# 📊 Executive Decision Report\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Run ID:** {run_id}\n")
        f.write(f"- **Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n")
        f.write("- **Framework Version:** Sreejita v3.5.1\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")

        f.write(
            "\n> This report presents decision-grade insights generated using "
            "**Sreejita Composite Intelligence**, designed for executive action.\n\n"
        )

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

        # INSIGHTS
        f.write("### 🧠 Strategic Insights\n")
        if insights:
            for ins in insights:
                f.write(
                    f"#### {self._level_icon(ins.get('level'))} {ins.get('title')}\n"
                )
                f.write(f"{ins.get('so_what')}\n\n")
        else:
            f.write("_Operations within expected parameters._\n\n")

        # KPIs
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in list(kpis.items())[:10]:
                f.write(
                    f"| {k.replace('_', ' ').title()} | "
                    f"**{self._format_value(k, v)}** |\n"
                )
            f.write("\n")

        # VISUALS
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for idx, vis in enumerate(visuals[:4], start=1):
                path = vis.get("path")
                if not path:
                    continue
                caption = vis.get("caption", "Visualization")
                img = f"visuals/{Path(path).name}"
                f.write(f"![{caption}]({img})\n")
                f.write(f"> *Fig {idx}.1 — {caption}*\n\n")

        # RECOMMENDATION SNAPSHOT
        if recs:
            primary = recs[0]
            f.write("### 🚀 Recommendation Snapshot\n")
            f.write(
                f"- **Action:** {primary.get('action')}\n"
                f"- **Priority:** {primary.get('priority', 'HIGH')}\n"
                f"- **Timeline:** {primary.get('timeline', 'Immediate')}\n\n"
            )

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------
    def _write_footer(self, f):
        f.write("\n---\n")
        f.write(
            "_Prepared by **Sreejita Data Labs** · "
            "Framework v3.5.1 · Confidential_\n"
        )

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
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
            if any(x in key.lower() for x in ["rate", "ratio", "margin"]) and abs_v <= 2:
                return f"{v:.1%}"
            if abs_v >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"
            if abs_v >= 1_000:
                return f"{v / 1_000:.1f}K"
            return f"{v:,}"
        return str(v)


# =====================================================
# STABLE ENTRY POINT (v3.5.1)
# =====================================================
def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    v3.5.1 FINAL CONTRACT
    """
    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Domain results
    domain_results = generate_report_payload(input_path, config)

    # 2️⃣ Markdown
    engine = HybridReport()
    md_path = engine.build(
        domain_results,
        run_dir,
        config.get("metadata"),
        config,
    )

    # 3️⃣ Narrative for PDF (DETERMINISTIC)
    primary_domain = engine._sort_domains(domain_results.keys())[0]
    result = domain_results.get(primary_domain, {})

    narrative = generate_narrative(result, config)

    payload = {
        "meta": {
            "domain": primary_domain.replace("_", " ").title(),
        },
        "summary": [narrative["executive_summary"]],
        "narrative": narrative,  # 🔥 FUTURE-PROOF
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
