import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
# KPI Formatting (Contract-Driven)
# =====================================================
def format_kpi_value(kpi_name, value):
    contract = KPI_REGISTRY.get(kpi_name)

    if value is None:
        return "N/A"

    if not contract:
        return str(value)

    if contract.unit == "currency":
        return f"${value:,.0f}"

    if contract.unit == "percent":
        return f"{value:.1f}%"

    if contract.unit == "count":
        return f"{int(value):,}"

    return str(value)


# =====================================================
# Executive Summary (Narrative-Driven)
# =====================================================
def render_executive_brief(story, styles, payload):
    kpis = payload["kpis"]
    insights = payload["insights"]
    narrative = payload.get("narrative", {})

    warnings = sum(1 for i in insights if i.get("level") == "WARNING")
    risks = sum(1 for i in insights if i.get("level") == "RISK")

    box = ParagraphStyle(
        "exec_box",
        parent=styles["BodyText"],
        backColor="#F2F4F7",
        borderPadding=10,
        spaceAfter=14,
    )

    story.append(Paragraph("<b>EXECUTIVE BRIEF (1-MINUTE READ)</b>", box))

    # Headline KPI
    headline = narrative.get("headline")
    if headline:
        kpi_key = headline.get("kpi")
        label = headline.get("label", "Key Metric")

        value = format_kpi_value(kpi_key, kpis.get(kpi_key))
        story.append(Paragraph(f"{label}: {value}", box))

    story.append(
        Paragraph(
            f"⚠️ Issues Identified: {warnings} WARNING(s), {risks} RISK(s)",
            box,
        )
    )

    # Semantic warnings surfaced
    semantic_flags = [
        i for i in insights if "semantic_warning" in i
    ]
    if semantic_flags:
        story.append(
            Paragraph(
                f"⚠️ {len(semantic_flags)} insight(s) require semantic review",
                box,
            )
        )

    next_step = narrative.get("default_next_step")
    if next_step:
        story.append(Paragraph(f"🎯 Recommended Next Step: {next_step}", box))

    story.append(Spacer(1, 12))


# =====================================================
# Main Entry
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    df_raw = pd.read_csv(input_path, encoding="latin1")
    df = clean_dataframe(df_raw)["df"]

    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy)
    if payload is None:
        raise RuntimeError("Report payload generation failed")

    kpis = payload["kpis"]
    insights = payload["insights"]
    visuals = payload["visuals"]

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], alignment=1)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # ================= EXECUTIVE =================
    render_executive_brief(story, styles, payload)

    # ================= KPIs =================
    story.append(PageBreak())
    story.append(Paragraph("Key Performance Indicators", title))
    story.append(Spacer(1, 12))

    table_data = []
    for k, v in kpis.items():
        label = k.replace("_", " ").title()
        table_data.append([label, format_kpi_value(k, v)])

    table = Table(table_data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
            ("BACKGROUND", (0, 0), (-1, 0), "#F2F4F7"),
        ])
    )
    story.append(table)

    # ================= VISUALS =================
    if visuals:
        story.append(PageBreak())
        story.append(Paragraph("Visual Evidence", title))
        for v in visuals:
            story.append(Image(str(v["path"]), width=14 * cm, height=8 * cm))
            story.append(Paragraph(v.get("caption", ""), styles["BodyText"]))
            story.append(Spacer(1, 14))

    # ================= INSIGHTS =================
    story.append(PageBreak())
    story.append(Paragraph("Key Insights", title))
    for ins in insights:
        badge = ins["level"]
        text = f"[{badge}] {ins['title']} — {ins.get('value','')}"
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Paragraph(ins.get("why", ""), styles["BodyText"]))
        story.append(Paragraph(ins.get("so_what", ""), styles["BodyText"]))

        if "semantic_warning" in ins:
            story.append(
                Paragraph(
                    f"⚠️ {ins['semantic_warning']}",
                    styles["Italic"],
                )
            )
        story.append(Spacer(1, 10))

    doc.build(story)
    return str(output_path)
