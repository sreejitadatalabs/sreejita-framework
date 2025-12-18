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
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        return f"${value:,.2f}"

    if contract.unit == "percent":
        return f"{value:.1f}%"

    if contract.unit == "count":
        return f"{int(value):,}"

    return str(value)


# =====================================================
# FULL REPORT
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Full_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    # -------------------------
    # Load & Prepare Data
    # -------------------------
    df_raw = pd.read_csv(input_path, encoding="latin1")
    df = clean_dataframe(df_raw)["df"]

    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy)
    if payload is None:
        raise RuntimeError("Report payload generation failed")

    kpis = payload.get("kpis", {})
    insights = payload.get("insights", [])
    recommendations = payload.get("recommendations", [])
    visuals = payload.get("visuals", [])
    narrative = payload.get("narrative", {})

    warnings = sum(1 for i in insights if i.get("level") == "WARNING")
    risks = sum(1 for i in insights if i.get("level") == "RISK")

    # -------------------------
    # PDF Setup
    # -------------------------
    styles = getSampleStyleSheet()

    box = ParagraphStyle(
        "box",
        parent=styles["BodyText"],
        backColor="#F2F4F7",
        borderPadding=10,
        spaceAfter=12,
    )

    heading = ParagraphStyle(
        "heading",
        parent=styles["Heading3"],
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # =====================================================
    # PAGE 1 — EXECUTIVE SNAPSHOT
    # =====================================================
    story.append(Paragraph("<b>EXECUTIVE BRIEF (1-MINUTE READ)</b>", box))

    headline = narrative.get("headline", {})
    if headline:
        kpi_key = headline.get("kpi")
        label = headline.get("label", "Key Metric")
        story.append(
            Paragraph(
                f"■ {label}: {format_kpi_value(kpi_key, kpis.get(kpi_key))}",
                box,
            )
        )

    story.append(
        Paragraph(
            f"■ Issues Identified: {warnings} WARNING(s), {risks} RISK(s)",
            box,
        )
    )

    next_step = narrative.get("default_next_step")
    if next_step:
        story.append(Paragraph(f"■ Next Step: {next_step}", box))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Executive Snapshot", heading))

    snapshot_data = [
        ["Detected Domain", decision.selected_domain],
        ["Confidence Score", f"{decision.confidence:.2f}"],
        ["Policy Status", policy.status],
    ]

    snapshot_table = Table(snapshot_data, colWidths=[6 * cm, 8 * cm])
    snapshot_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
            ("BACKGROUND", (0, 0), (-1, 0), "#F2F4F7"),
        ])
    )
    story.append(snapshot_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Key Performance Indicators", heading))

    kpi_rows = [
        [k.replace("_", " ").title(), format_kpi_value(k, v)]
        for k, v in kpis.items()
    ]

    kpi_table = Table(kpi_rows, colWidths=[8 * cm, 6 * cm])
    kpi_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
            ("BACKGROUND", (0, 0), (-1, 0), "#F2F4F7"),
        ])
    )
    story.append(kpi_table)

    # =====================================================
    # PAGE 2 — VISUAL EVIDENCE
    # =====================================================
    if visuals:
        story.append(PageBreak())
        story.append(Paragraph("Visual Evidence", heading))
        story.append(Spacer(1, 12))

        for v in visuals:
            story.append(Image(str(v["path"]), width=14 * cm, height=8 * cm))
            if v.get("caption"):
                story.append(Paragraph(v["caption"], styles["BodyText"]))
            story.append(Spacer(1, 16))

    # =====================================================
    # PAGE 3 — INSIGHTS + RECOMMENDATIONS
    # =====================================================
    story.append(PageBreak())
    story.append(Paragraph("Key Insights & Recommended Actions", heading))
    story.append(Spacer(1, 12))

    for idx, ins in enumerate(insights, start=1):
        story.append(
            Paragraph(
                f"<b>{idx}. [{ins['level']}] {ins['title']}</b>",
                styles["BodyText"],
            )
        )
        story.append(Paragraph(ins.get("why", ""), styles["BodyText"]))
        story.append(Paragraph(ins.get("so_what", ""), styles["BodyText"]))

        if "semantic_warning" in ins:
            story.append(
                Paragraph(
                    f"⚠ {ins['semantic_warning']}",
                    styles["Italic"],
                )
            )

        # Attach recommendations immediately after insights
        for rec in recommendations:
            story.append(
                Paragraph(
                    f"→ <b>Action:</b> {rec.get('action')} "
                    f"(Priority: {rec.get('priority','MEDIUM')})",
                    styles["BodyText"],
                )
            )
            if rec.get("expected_impact"):
                story.append(
                    Paragraph(
                        f"Expected Impact: {rec['expected_impact']}",
                        styles["BodyText"],
                    )
                )
            if rec.get("timeline"):
                story.append(
                    Paragraph(
                        f"Timeline: {rec['timeline']}",
                        styles["BodyText"],
                    )
                )
            story.append(Spacer(1, 10))

    doc.build(story)
    return str(output_path)
