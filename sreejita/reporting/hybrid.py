from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from dataclasses import asdict

from sreejita.reporting.base import BaseReport
from sreejita.narrative.engine import build_narrative


# =====================================================
# HYBRID REPORT ENGINE (v3.6.5 — PLATINUM)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Report Engine
    Features: 
    - Board Confidence & Maturity Injection
    - Executive Narrative Integration
    - Deterministic Rendering
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENGINE ENTRY POINT
    # -------------------------------------------------
    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        narrative_data: Any,
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
            self._write_narrative(f, narrative_data)
            
            # Sort domains to show Healthcare/Finance first
            sorted_domains = self._sort_domains(domain_results.keys())
            
            for domain in sorted_domains:
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # EXECUTIVE NARRATIVE
    # -------------------------------------------------
    def _write_narrative(self, f, narrative):
        f.write("\n## 1. Executive Brief\n\n")

        # Executive Summary
        summary = getattr(narrative, "executive_summary", []) or []
        if not summary:
            summary = ["Operational performance indicators require management attention."]
        
        for line in summary:
            f.write(f"  • {line}\n")
        f.write("\n")

        # Financial Impact (The "So What?")
        financial = getattr(narrative, "financial_impact", []) or []
        if financial:
            f.write("### 💰 Financial Impact\n")
            for line in financial:
                f.write(f"- {line}\n")
            f.write("\n")

        # Strategic Risks
        risks = getattr(narrative, "risks", []) or []
        if risks:
            f.write("### ⚠️ Strategic Risks\n")
            for line in risks:
                f.write(f"- {line}\n")
            f.write("\n")

        # Action Plan Table
        actions = getattr(narrative, "action_plan", []) or []
        if actions:
            f.write("### 🚀 Strategic Action Plan\n")
            f.write("| Action | Owner | Timeline | Success Metric |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for item in actions:
                # Handle both dict (from JSON) and dataclass (from engine)
                act = item if isinstance(item, dict) else asdict(item)
                f.write(f"| {act.get('action')} | {act.get('owner')} | {act.get('timeline')} | {act.get('success_kpi')} |\n")
            f.write("\n")

        f.write("---\n\n")

    # -------------------------------------------------
    # DOMAIN DEEP DIVE (With Board Intelligence)
    # -------------------------------------------------
    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write(f"## 2. Deep Dive: {domain.replace('_', ' ').title()}\n\n")

        if not isinstance(result, dict):
            f.write("_Invalid domain output._\n\n")
            return

        kpis = result.get("kpis", {}) or {}
        visuals = result.get("visuals", []) or []

        # --- 🧭 BOARD CONFIDENCE SCORE INJECTION ---
        score = kpis.get("board_confidence_score")
        trend = kpis.get("board_confidence_trend", "→")
        
        if isinstance(score, (int, float)):
            f.write("### 🧭 Board Confidence Score\n\n")

            # Logic: Determine Label & Interpretation
            if score >= 85:
                label = "🟢 High Confidence"
                note = "Operations are stable with no immediate executive intervention required."
            elif score >= 70:
                label = "🟡 Moderate Confidence"
                note = "Targeted operational improvements recommended."
            elif score >= 50:
                label = "🟠 Elevated Risk"
                note = "Leadership attention required to prevent degradation."
            else:
                label = "🔴 Critical Risk"
                note = "Immediate executive intervention required."

            # Logic: Trend Text
            trend_label = {
                "↑": "Improving",
                "→": "Stable",
                "↓": "Deteriorating"
            }.get(trend, "Stable")

            # Write Block
            f.write(f"- **Score:** **{score} / 100** ({label})\n")
            f.write(f"- **Trend:** {trend} **{trend_label}**\n")
            f.write(f"- **Interpretation:** {note}\n")
            
            # 🏅 Maturity Level Injection
            maturity = kpis.get("maturity_level")
            if maturity:
                icon = {
                    "Gold": "🥇",
                    "Silver": "🥈",
                    "Bronze": "🥉"
                }.get(maturity, "ℹ️")
                f.write(f"- **Maturity Level:** {icon} **{maturity}**\n\n")
            else:
                f.write("\n")

        # --- 📊 OPERATIONAL METRICS (Filtered) ---
        if kpis:
            f.write("### 📊 Operational Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            
            # Hide internal meta-keys from the clean table
            hidden_keys = [
                "board_confidence_score", 
                "maturity_level", 
                "board_confidence_trend", 
                "dataset_shape", 
                "is_aggregated",
                "debug_shape_score",
                "care_context",
                "board_confidence_explanation" # Added this so it doesn't break table
            ]
            
            # Limit to top 12 metrics to prevent spam
            for k, v in list(kpis.items())[:12]:
                if k not in hidden_keys:
                    f.write(f"| {k.replace('_',' ').title()} | **{self._format_value(k, v)}** |\n")
            f.write("\n")

        # --- 👁️ VISUAL EVIDENCE ---
        if visuals:
            f.write("### 📉 Visual Evidence\n")
            # Strict Limit: Top 6 Visuals Only
            for vis in visuals[:6]:
                img_path = vis.get("path", "")
                if img_path and "placeholder" not in img_path:
                    # Markdown image link
                    f.write(f"![{vis.get('caption')}]({img_path})\n")
                    f.write(f"> *{vis.get('caption')}*\n\n")

    # -------------------------------------------------
    # HEADER & FOOTER
    # -------------------------------------------------
    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write(f"# Sreejita Executive Report\n\n")
        f.write(f"**Run ID:** `{run_id}` | **Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_',' ').title()}**: {v}\n")
        f.write("---\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Generated by **Sreejita Intelligence Engine** · Framework v3.6.5_\n")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def _sort_domains(self, domains):
        priority = ["healthcare", "finance", "sales", "marketing"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _format_value(self, key: str, v: Any):
        if isinstance(v, (int, float)):
            if any(x in key.lower() for x in ["rate", "ratio", "margin", "ctr", "roas"]):
                return f"{v:.1%}" if abs(v) <= 5 else f"{v:.2f}"
            if abs(v) >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"
            if abs(v) >= 1_000:
                return f"{v / 1_000:.1f}K"
            return f"{v:,.0f}" if isinstance(v, (int)) or v > 10 else f"{v:.2f}"
        return str(v)


# =====================================================
# PUBLIC ENTRY POINT
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrator Entry Point.
    Ties together Domain Logic -> Narrative -> Markdown Generation.
    """
    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute Domain Intelligence
    domain_results = generate_report_payload(input_path, config)

    # 2. Identify Primary Domain
    engine = HybridReport()
    primary_domain = engine._sort_domains(domain_results.keys())[0]
    result = domain_results.get(primary_domain, {}) or {}

    # 3. Generate Narrative (With Financial Translation)
    narrative = build_narrative(
        domain=primary_domain,
        kpis=result.get("kpis", {}),
        insights=result.get("insights", []),
        recommendations=result.get("recommendations", []),
    )

    # 4. Build Markdown Report
    md_path = engine.build(
        domain_results=domain_results,
        narrative_data=narrative,
        output_dir=run_dir,
        metadata=config.get("metadata"),
        config=config,
    )

    # 5. Construct Payload (UI/PDF Safe)
    payload = {
        "meta": {
            "domain": primary_domain.replace("_", " ").title(),
            "run_id": f"RUN-{datetime.utcnow():%Y%m%d-%H%M%S}",
        },
        "summary": narrative.executive_summary,
        "kpis": result.get("kpis", {}),
        "visuals": result.get("visuals", []),
        "insights": narrative.key_findings,
        # Ensure Recommendations are dicts for JSON serialization
        "recommendations": [asdict(r) if not isinstance(r, dict) else r for r in narrative.action_plan],
        "risks": narrative.risks,
        "financial_impact": narrative.financial_impact,
    }

    return {
        "markdown": str(md_path),
        "payload": payload,
        "run_dir": str(run_dir),
    }
