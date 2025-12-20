import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sreejita.reporting.base import BaseReport

# =====================================================
# HYBRID REPORT (v3.1 - ENTERPRISE DECISION ENGINE)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.1 Report Engine
    - Decision-first Narrative
    - Visual Evidence Integration
    - Composite Authority Logic
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENTRY POINT
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Dict[str, Any] | None = None
    ) -> Path:

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamped filename
        filename = f"Sreejita_Executive_Report_{datetime.now():%Y-%m-%d}.md"
        report_path = output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, metadata)

            # Sort domains to put critical ones (Finance, Retail) first if present
            ordered_domains = self._sort_domains(domain_results.keys())

            for domain in ordered_domains:
                self._write_domain_section(f, domain, domain_results[domain])

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # SECTIONS
    # -------------------------------------------------

    def _write_header(self, f, metadata: Dict[str, Any] | None):
        f.write(f"# 📊 Executive Decision Report\n")
        f.write(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}\n\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write(
            "> ⚠️ **Executive Summary**: This report uses **v3.0 Composite Intelligence** to prioritize "
            "root-cause risks over raw metrics. Actions listed below are authoritative mandates.\n\n"
        )

    def _write_domain_section(
        self,
        f,
        domain: str,
        result: Dict[str, Any]
    ):
        f.write(f"---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        kpis = result.get("kpis", {})
        insights = result.get("insights", [])
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # 1️⃣ Strategic Insights (The "Why")
        # We put insights FIRST in v3.0 because they drive the narrative.
        authoritative_insights = self._prioritize_insights(insights)
        
        if authoritative_insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for ins in authoritative_insights:
                icon = self._level_icon(ins["level"])
                f.write(f"#### {icon} {ins['title']}\n")
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("✅ _Operations within normal parameters._\n\n")

        # 2️⃣ Key Metrics (The "What" - as a Table)
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            
            # Filter out internal/debug keys if any
            clean_kpis = {k: v for k, v in kpis.items() if not k.startswith("_")}
            
            # Limit to top 8 KPIs to keep it executive
            for k, v in list(clean_kpis.items())[:8]: 
                fmt_val = self._format_value(v)
                label = k.replace('_', ' ').title().replace("Kpi", "KPI")
                f.write(f"| {label} | **{fmt_val}** |\n")
            f.write("\n")

        # 3️⃣ Visual Evidence (The "Proof")
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            # Show max 2 visuals to keep report tight
            for vis in visuals[:2]:
                path_obj = Path(vis['path'])
                # Markdown requires just the filename if in same folder
                rel_path = path_obj.name 
                caption = vis.get('caption', 'Data Visualization')
                f.write(f"![{caption}]({rel_path})\n")
                f.write(f"> *{caption}*\n\n")

        # 4️⃣ Authoritative Actions (The "How")
        if recs:
            f.write("### 🚀 Required Actions\n")
            
            # Sort by priority
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            sorted_recs = sorted(
                recs, 
                key=lambda r: priority_order.get(r.get("priority", "LOW"), 3)
            )

            # Primary Mandate
            primary = sorted_recs[0]
            f.write(f"**PRIMARY MANDATE:** {primary['action']}\n")
            f.write(f"- 🚨 **Priority:** {primary.get('priority', 'HIGH')}\n")
            f.write(f"- ⏱️ **Timeline:** {primary.get('timeline', 'Immediate')}\n\n")

            # Secondary Actions (if any)
            if len(sorted_recs) > 1:
                f.write("**Supporting Actions:**\n")
                for r in sorted_recs[1:3]: # Max 2 secondary
                    f.write(f"- {r['action']} ({r.get('timeline', 'Ongoing')})\n")
            f.write("\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Powered by Sreejita Framework v3.1 | Enterprise Decision Engine_")

    # -------------------------------------------------
    # INTELLIGENCE LOGIC
    # -------------------------------------------------

    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts insights by severity: RISK > WARNING > INFO.
        Domains have already suppressed atomic noise, so we just order for impact.
        """
        if not insights:
            return []

        def rank(i):
            return {"RISK": 0, "WARNING": 1, "INFO": 2}.get(i["level"], 3)

        return sorted(insights, key=rank)[:5]

    def _sort_domains(self, domains):
        """Ensures Finance and Retail appear at the top."""
        priority = ["finance", "retail", "ecommerce", "supply_chain"]
        
        def sorter(d):
            if d in priority:
                return priority.index(d)
            return 99
            
        return sorted(domains, key=sorter)

    def _level_icon(self, level: str) -> str:
        return {
            "RISK": "🔴",
            "WARNING": "🟠",
            "INFO": "🔵"
        }.get(level, "ℹ️")

    def _format_value(self, v: Any) -> str:
        """Helper to format numbers nicely."""
        if isinstance(v, float):
            # If it looks like a percentage (0.0 - 1.0), format as %
            if 0 < abs(v) < 1.0: 
                return f"{v:.1%}"
            # Otherwise comma separator
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

