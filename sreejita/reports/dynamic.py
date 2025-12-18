import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpi_normalizer import KPI_REGISTRY


# =====================================================
# Helpers
# =====================================================
def format_kpi_value(kpi_name, value):
    contract = KPI_REGISTRY.get(kpi_name)
    if value is None:
        return "N/A"
    if not contract:
        return str(value)

    if contract.unit == "currency":
        return f"${value:,.2f}"
    if contract.unit == "percent":
        return f"{value:.1f}%"
    if contract.unit == "count":
        return f"{int(value):,}"
    return str(value)


def filter_dict(data: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in data.items() if k in keys}


# =====================================================
# Dynamic Report
# =====================================================
def run(
    input_path: str,
    config: dict,
    output_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generates:
    - Dynamic PDF report (human-readable)
    - Dynamic JSON payload (machine-readable)
    """

    input_path = Path(input_path)

    out_dir = input_path.parent / "reports"
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    pdf_path = out_dir / f"Dynamic_Report_{timestamp}.pdf"
    json_path = out_dir / f"Dynamic_Report_{timestamp}.json"

    # -------------------------
    # Load & Analyze
    # -------------------------
    df_raw = pd.read_csv(input_path, encoding="latin1")
    df = clean_dataframe(df_raw)["df"]

    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy)
    if payload is None:
        raise RuntimeError("Dynamic report payload generation failed")

    # -------------------------
    # JSON OUTPUT (STRICT)
    # -------------------------
    json_output: Dict[str, Any] = {
        "generated_at": timestamp,
        "report_type": "dynamic",
        "domain": decision.selected_domain,
        "confidence": decision.confidence,
        "policy_status": policy.status,
        "sections": {},
    }

    sections = config.get("sections", [])

    # -------------------------
    # OVERVIEW
    # -------------------------
    if "overview" in sections:
        overview_cfg = config.get("overview", {})
        overview = {}

        if "dataset_health" in overview_cfg.get("include", []):
            overview["rows"] = len(df)
            overview["columns"] = list(df.columns)

        if "domain_decision" in overview_cfg.get("include", []):
            overview["domain"] = decision.selected_domain
            overview["confidence"] = decision.confidence

        if "policy_status" in overview_cfg.get("include", []):
            overview["policy_status"] = policy.status

        json_output["sections"]["overview"] = overview

    # -------------------------
    # KPIs
    # -------------------------
    if "kpis" in sections:
        kpi_cfg = config.get("kpis", {})
        selected = kpi_cfg.get("include", [])
        kpis = filter_dict(payload.get("kpis", {}), selected)
        json_output["sections"]["kpis"] = kpis

    # -------------------------
    # INSIGHTS
    # -------------------------
    if "insights" in sections:
        ins_cfg = config.get("insights", {})
        levels = ins_cfg.get("include_levels", [])
        limit = ins_cfg.get("limit")

        insights = [
            i for i in payload.get("insights", [])
            if not levels or i.get("level") in levels
        ]
        if limit:
            insights = insights[:limit]

        json_output["sections"]["insights"] = insights

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    if "recommendations" in sections:
        rec_cfg = config.get("recommendations", {})
        limit = rec_cfg.get("limit")

        recs = payload.get("recommendations", [])
        if limit:
            recs = recs[:limit]

        json_output["sections"]["recommendations"] = recs

    # -------------------------
    # VISUALS
    # -------------------------
    visuals = []
    if "visuals" in sections:
        vis_cfg = config.get("visuals", {})
        limit = vis_cfg.get("limit")

        visuals = payload.get("visuals", [])
        if limit:
            visuals = visuals[:limit]

        json_output["sections"]["visuals"] = [
            {"path": str(v["path"]), "caption": v.get("caption", "")}
            for v in visuals
        ]

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    # =====================================================
    # PDF OUTPUT (STRICT, MIRRORS CONFIG)
    # =====================================================
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    title = config.get("report", {}).get("title", "Dynamic Report")
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 12))

    # Render sections in exact order
    for section in sections:

        story.append(Spacer(1, 10))
        story.append(Paragraph(section.title(), styles["Heading2"]))
        story.append(Spacer(1, 8))

        data = json_output["sections"].get(section, {})

        if section == "overview":
            for k, v in data.items():
                story.append(Paragraph(f"<b>{k}:</b> {v}", styles["BodyText"]))

        elif section == "kpis":
            rows = [
                [k.replace("_", " ").title(), format_kpi_value(k, v)]
                for k, v in data.items()
            ]
            table = Table(rows, colWidths=[8 * cm, 6 * cm])
            table.setStyle(
                TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
                ])
            )
            story.append(table)

        elif section == "insights":
            for i in data:
                story.append(
                    Paragraph(
                        f"[{i.get('level')}] {i.get('title')}",
                        styles["BodyText"],
                    )
                )

        elif section == "recommendations":
            for r in data:
                story.append(
                    Paragraph(
                        f"→ {r.get('action')}",
                        styles["BodyText"],
                    )
                )

        elif section == "visuals":
            for v in visuals:
                story.append(Image(v["path"], width=14 * cm, height=8 * cm))
                if v.get("caption"):
                    story.append(Paragraph(v["caption"], styles["BodyText"]))

        story.append(PageBreak())

    doc.build(story)

    return {
        "pdf": str(pdf_path),
        "json": str(json_path),
    }
