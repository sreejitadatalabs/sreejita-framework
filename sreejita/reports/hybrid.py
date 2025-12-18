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
)

from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpi_normalizer import KPI_REGISTRY


# =====================================================
# KPI Formatting (STRICTLY CONTRACT-DRIVEN)
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
# EXECUTIVE PAGE-1 REPORT
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    # -------------------------
    # Load & Clean Data
    # -------------------------
    df_raw = pd.read_csv(input_path, encoding="latin1")
    df = clean_dataframe(df_raw)["df"]

    # -------------------------
    # Decision + Policy
    # -------------------------
    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    # -------------------------
    # Payload
    # -------------------------
    payload = generate_report_payload(df, decision, policy)
    if payload is None:
        raise RuntimeError("Report payload generation failed")

    kpis = payload.get("kpis", {})
    insights = payload.get("insights", [])
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
        spaceAfter=6,
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
    # EXECUTIVE BRIEF
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

    # =====================================================
    # EXECUTIVE SNAPSHOT
    # =====================================================
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

    # =====================================================
    # KEY PERFORMANCE INDICATORS
    # =====================================================
    story.append(Spacer(1, 14))
    story.append(Paragraph("Key Performance Indicators", heading))

    kpi_rows = []
    for kpi_name, value in kpis.items():
        label = kpi_name.replace("_", " ").title()
        kpi_rows.append([label, format_kpi_value(kpi_name, value)])

    kpi_table = Table(kpi_rows, colWidths=[8 * cm, 6 * cm])
    kpi_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
            ("BACKGROUND", (0, 0), (-1, 0), "#F2F4F7"),
        ])
    )
    story.append(kpi_table)

    # =====================================================
    # BUILD (PAGE-1 ONLY)
    # =====================================================
    doc.build(story)
    return str(output_path)
