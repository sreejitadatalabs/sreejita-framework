from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from dataclasses import asdict

from sreejita.reporting.base import BaseReport
from sreejita.narrative.engine import build_narrative

class HybridReport(BaseReport):
    """
    Hybrid Report Engine: Platinum Edition
    """
    name = "hybrid"

    def build(self, domain_results, narrative_data, output_dir, metadata=None, config=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "Sreejita_Executive_Report.md"
        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)
            self._write_narrative(f, narrative_data)
            sorted_domains = self._sort_domains(domain_results.keys())
            for domain in sorted_domains:
                self._write_domain_section(f, domain, domain_results.get(domain, {}))
            self._write_footer(f)
        return report_path

    def _write_narrative(self, f, narrative):
        f.write("\n## 1. Executive Brief\n\n")
        summary = getattr(narrative, "executive_summary", []) or ["Operational performance indicators require management attention."]
        for line in summary: f.write(f"  • {line}\n")
        f.write("\n")

        financial = getattr(narrative, "financial_impact", []) or []
        if financial:
            f.write("### 💰 Financial Impact\n")
            for line in financial: f.write(f"- {line}\n")
            f.write("\n")

        risks = getattr(narrative, "risks", []) or []
        if risks:
            f.write("### ⚠️ Strategic Risks\n")
            for line in risks: f.write(f"- {line}\n")
            f.write("\n")

        actions = getattr(narrative, "action_plan", []) or []
        if actions:
            f.write("### 🚀 Strategic Action Plan\n")
            f.write("| Action | Owner | Timeline | Success Metric |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for item in actions:
                act = item if isinstance(item, dict) else asdict(item)
                f.write(f"| {act.get('action')} | {act.get('owner')} | {act.get('timeline')} | {act.get('success_kpi')} |\n")
            f.write("\n")
        f.write("---\n\n")

    def _write_domain_section(self, f, domain, result):
        f.write(f"## 2. Deep Dive: {domain.replace('_', ' ').title()}\n\n")
        if not isinstance(result, dict):
            f.write("_Invalid domain output._\n\n")
            return

        kpis = result.get("kpis", {}) or {}
        visuals = result.get("visuals", []) or []

        # SCORE CARD
        score = kpis.get("board_confidence_score")
        if isinstance(score, (int, float)):
            f.write("### 🧭 Board Confidence Score\n\n")
            if score >= 85: label, note = "🟢 High Confidence", "Operations stable."
            elif score >= 70: label, note = "🟡 Moderate Confidence", "Targeted improvements needed."
            elif score >= 50: label, note = "🟠 Elevated Risk", "Leadership attention required."
            else: label, note = "🔴 Critical Risk", "Immediate intervention required."
            
            f.write(f"- **Score:** **{score} / 100** ({label})\n")
            f.write(f"- **Trend:** {kpis.get('board_confidence_trend', '→')} \n")
            f.write(f"- **Interpretation:** {note}\n")
            
            maturity = kpis.get("maturity_level")
            if maturity:
                icon = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉"}.get(maturity, "ℹ️")
                f.write(f"- **Maturity Level:** {icon} **{maturity}**\n")

            # 🔧 SCORE BREAKDOWN TABLE
            breakdown = kpis.get("board_score_breakdown", {})
            if breakdown:
                f.write("\n**Score Drivers:**\n")
                f.write("| Driver | Impact |\n| :--- | :---: |\n")
                for r, p in breakdown.items():
                    sign = "+" if p > 0 else ""
                    f.write(f"| {r} | **{sign}{p}** |\n")
            f.write("\n")

        # METRICS TABLE
        if kpis:
            f.write("### 📊 Operational Metrics\n")
            f.write("| Metric | Value |\n| :--- | :--- |\n")
            hidden = ["board_confidence_score", "maturity_level", "board_confidence_trend", "dataset_shape", "is_aggregated", "care_context", "board_confidence_explanation", "board_score_breakdown"]
            for k, v in list(kpis.items())[:12]:
                if k not in hidden:
                    f.write(f"| {k.replace('_',' ').title()} | **{self._format_value(k, v)}** |\n")
            f.write("\n")

        # VISUALS
        if visuals:
            f.write("### 📉 Visual Evidence\n")
            for vis in visuals[:6]:
                if vis.get("path") and "placeholder" not in vis.get("path"):
                    f.write(f"![{vis.get('caption')}]({vis.get('path')})\n")
                    f.write(f"> *{vis.get('caption')}*\n\n")

    def _write_header(self, f, run_id, metadata):
        f.write(f"# Sreejita Executive Report\n\n")
        f.write(f"**Run ID:** `{run_id}` | **Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n\n")
        if metadata:
            for k, v in metadata.items(): f.write(f"- **{k.replace('_',' ').title()}**: {v}\n")
        f.write("---\n")

    def _write_footer(self, f):
        f.write("\n---\n_Generated by **Sreejita Intelligence Engine** · Framework v3.7_\n")

    def _sort_domains(self, domains):
        priority = ["healthcare", "finance", "sales", "marketing"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _format_value(self, key, v):
        if isinstance(v, (int, float)):
            if any(x in key.lower() for x in ["rate", "ratio", "margin"]): return f"{v:.1%}" if abs(v) <= 5 else f"{v:.2f}"
            if abs(v) >= 1_000_000: return f"{v / 1_000_000:.1f}M"
            if abs(v) >= 1_000: return f"{v / 1_000:.1f}K"
            return f"{v:,.0f}" if isinstance(v, int) or v > 10 else f"{v:.2f}"
        return str(v)

def run(input_path, config):
    from sreejita.reporting.orchestrator import generate_report_payload
    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    domain_results = generate_report_payload(input_path, config)
    engine = HybridReport()
    primary = engine._sort_domains(domain_results.keys())[0]
    res = domain_results.get(primary, {})
    narrative = build_narrative(primary, res.get("kpis", {}), res.get("insights", []), res.get("recommendations", []))
    md_path = engine.build(domain_results, narrative, run_dir, config.get("metadata"), config)
    return {"markdown": str(md_path), "payload": {"meta": {"domain": primary}, "summary": narrative.executive_summary}, "run_dir": str(run_dir)}
