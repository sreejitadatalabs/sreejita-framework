import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# --- Internal Imports ---
from sreejita.reporting.base import BaseReport
from sreejita.core.cleaner import clean_dataframe
from sreejita.domains.router import decide_domain
from sreejita.domains import (
    finance, retail, ecommerce, supply_chain,
    marketing, customer, hr, healthcare
)

# =====================================================
# HELPERS (Data Loading)
# =====================================================

def robust_read_dataframe(path: Path) -> pd.DataFrame:
    """Robust loader for CSV and Excel files."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    elif suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

# =====================================================
# HYBRID REPORT (v3.2 - ROBUST DECISION ENGINE)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.2 Report Engine
    - Decision-first Narrative
    - Visual Evidence Integration
    - Composite Authority Logic
    - Context-Aware Formatting
    - Robustness Guards
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENTRY POINT
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:

        # Ensure output_dir is a Path object
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        elif isinstance(output_dir, dict):
            # Fallback if a config dict was passed by mistake, defaut to current dir
            output_dir = Path("reports")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped filename
        filename = f"Sreejita_Executive_Report_{datetime.now():%Y-%m-%d_%H%M}.md"
        report_path = output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, metadata)

            # Sort critical domains (Finance/Retail) to the top
            ordered_domains = self._sort_domains(list(domain_results.keys()))

            for domain in ordered_domains:
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {})
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # SECTIONS
    # -------------------------------------------------

    def _write_header(self, f, metadata: Optional[Dict[str, Any]]):
        f.write("# 📊 Executive Decision Report\n")
        f.write(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}\n\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write(
            "> ⚠️ **Executive Summary**: This report uses **v3.0 Composite Intelligence** "
            "to prioritize root-cause risks over raw metrics. "
            "Actions listed below are authoritative mandates.\n\n"
        )

    def _write_domain_section(
        self,
        f,
        domain: str,
        result: Dict[str, Any]
    ):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        kpis = result.get("kpis", {})
        insights = result.get("insights", [])
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # 1️⃣ Strategic Intelligence
        authoritative_insights = self._prioritize_insights(insights)

        if authoritative_insights:
            f.write("### 🧠 Strategic Intelligence\n")
            for ins in authoritative_insights:
                # Robustness Guard
                if "title" not in ins or "so_what" not in ins:
                    continue

                icon = self._level_icon(ins.get("level", "INFO"))
                f.write(f"#### {icon} {ins['title']}\n")
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("✅ _Operations within normal parameters._\n\n")

        # 2️⃣ Key Metrics
        if kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            clean_kpis = {k: v for k, v in kpis.items() if not k.startswith("_")}

            for k, v in list(clean_kpis.items())[:8]:
                label = k.replace("_", " ").title().replace("Kpi", "KPI")
                formatted_v = self._format_value(k, v)
                f.write(f"| {label} | **{formatted_v}** |\n")

            f.write("\n")

        # 3️⃣ Visual Evidence
        if visuals:
            f.write("### 👁️ Visual Evidence\n")
            for vis in visuals[:2]:
                img_path = Path(vis["path"])
                # If image is absolute, try to make it relative or use name
                img_ref = img_path.name
                caption = vis.get("caption", "Data Visualization")
                f.write(f"![{caption}]({img_ref})\n")
                f.write(f"> *{caption}*\n\n")

        # 4️⃣ Required Actions
        if recs:
            f.write("### 🚀 Required Actions\n")

            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            sorted_recs = sorted(
                recs,
                key=lambda r: priority_order.get(r.get("priority", "LOW"), 3)
            )

            primary = sorted_recs[0]
            f.write(f"**PRIMARY MANDATE:** {primary['action']}\n")
            f.write(f"- 🚨 **Priority:** {primary.get('priority', 'HIGH')}\n")
            f.write(f"- ⏱️ **Timeline:** {primary.get('timeline', 'Immediate')}\n\n")

            if len(sorted_recs) > 1:
                f.write("**Supporting Actions:**\n")
                for r in sorted_recs[1:3]:
                    f.write(f"- {r['action']} ({r.get('timeline', 'Ongoing')})\n")
                f.write("\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Powered by Sreejita Framework v3.1 | Enterprise Decision Engine_")

    # -------------------------------------------------
    # INTELLIGENCE LOGIC
    # -------------------------------------------------

    def _prioritize_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not insights:
            return []
        def rank(i):
            return {"RISK": 0, "WARNING": 1, "INFO": 2}.get(i.get("level"), 3)
        return sorted(insights, key=rank)[:5]

    def _sort_domains(self, domains: List[str]) -> List[str]:
        priority = ["finance", "retail", "ecommerce", "supply_chain", "marketing", "hr"]
        def sorter(d):
            return priority.index(d) if d in priority else 99
        return sorted(domains, key=sorter)

    def _level_icon(self, level: str) -> str:
        return {
            "RISK": "🔴",
            "WARNING": "🟠",
            "INFO": "🔵"
        }.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any) -> str:
        if isinstance(v, float):
            key_lower = key.lower()
            is_percentage_key = any(x in key_lower for x in [
                "rate", "ratio", "margin", "percent", "pct", "share", "bounce", "conversion"
            ])
            # Only treat small floats (<= 2.0) as percentages if the key suggests it
            if is_percentage_key and abs(v) <= 2.0: 
                return f"{v:.1%}"
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)


# =====================================================
# ORCHESTRATION ADAPTER (FIXES THE ERROR)
# =====================================================

def run(input_path: Union[str, Path], config: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Orchestrates the full analysis pipeline to satisfy legacy CLI calls.
    Args:
        input_path: Path to raw data file.
        config: Configuration dictionary (can contain output_dir).
        output_path: Optional override for output location.
    Returns:
        str: The path to the generated report file.
    """
    
    # 1. Setup Paths
    input_file = Path(input_path)
    
    # Determine output directory
    if output_path:
        out_dir = Path(output_path)
        if not out_dir.is_dir(): 
            out_dir = out_dir.parent # Ensure we have a directory
    else:
        # Fallback to a 'reports' folder next to input or from config
        out_dir = Path(config.get("output_dir", input_file.parent / "reports"))
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load & Clean Data
    print(f"🔄 Loading data from {input_file.name}...")
    df_raw = robust_read_dataframe(input_file)
    clean_result = clean_dataframe(df_raw)
    df = clean_result["df"]

    # 3. Detect Domain
    print("🧠 Detecting business domain...")
    decision = decide_domain(df)
    domain_name = decision.domain
    print(f"✅ Domain Detected: {domain_name.upper()} (Confidence: {decision.confidence:.2f})")

    # 4. Select Domain Engine
    domain_map = {
        "finance": finance.FinanceDomain,
        "retail": retail.RetailDomain,
        "ecommerce": ecommerce.EcommerceDomain,
        "supply_chain": supply_chain.SupplyChainDomain,
        "marketing": marketing.MarketingDomain,
        "customer": customer.CustomerDomain,
        "hr": hr.HRDomain,
        "healthcare": healthcare.HealthcareDomain
    }

    EngineClass = domain_map.get(domain_name)
    if not EngineClass:
        raise ValueError(f"No engine found for domain: {domain_name}")

    # 5. Execute Analysis
    engine = EngineClass()
    
    # Domain preprocessing
    df_proc = engine.preprocess(df)
    
    # Generate Intelligence
    kpis = engine.calculate_kpis(df_proc)
    visuals = engine.generate_visuals(df_proc, out_dir / "visuals")
    insights = engine.generate_insights(df_proc, kpis)
    recs = engine.generate_recommendations(df_proc, kpis)

    # 6. Structure Results
    domain_results = {
        domain_name: {
            "kpis": kpis,
            "insights": insights,
            "recommendations": recs,
            "visuals": visuals
        }
    }

    # 7. Generate Report
    print("📝 Generaing Executive Report...")
    report_engine = HybridReport()
    final_report_path = report_engine.build(
        domain_results=domain_results,
        output_dir=out_dir,
        metadata={
            "Source File": input_file.name,
            "Detected Domain": domain_name.title(),
            "Confidence": f"{decision.confidence:.1%}",
            "Rows Analyzed": f"{len(df):,}"
        }
    )

    return str(final_report_path)
