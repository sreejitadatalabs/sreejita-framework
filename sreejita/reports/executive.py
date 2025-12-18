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
# EXECUTIVE REPORT
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Executive_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

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

    # -------------------------
    # Selection Logic
    # -------------------------
    key_visual = visuals[0] if visuals else None
    top_recommendation = recommendations[0] if recommendations else None
    risks = [i for i in insights if i.get("level") == "RISK"][:3]

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

    h1 = ParagraphStyle("h1", parent=styles["Heading1"])
    body = styles["BodyText"]

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
    # PAGE 1 — EXECUTIVE SUMMARY
    # =====================================================
    story.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", box))

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
            f"■ Detected Domain: {decision.selected_domain} "
            f"(Confidence: {decision.confidence:.2f})",
            box,
        )
    )

    story.append(
        Paragraph(
            f"■ Policy Status: {policy.status}",
            box,
        )
    )

    next_step = narrative.get("default_next_step")
    if next_step:
        story.append(Paragraph(f"■ Immediate Focus: {next_step}", box))

    # =====================================================
    # PAGE 2 — EVIDENCE & DIRECTION
    # =====================================================
    story.append(PageBreak())
    story.append(Paragraph("Evidence & Direction", h1))
    story.append(Spacer(1, 10))

    if key_visual:
        story.append(Image(str(key_visual["path"]), width=14 * cm, height=8 * cm))
        if key_visual.get("caption"):
            story.append(Paragraph(key_visual["caption"], body))

    if top_recommendation:
        story.append(Spacer(1, 14))
        story.append(
            Paragraph(
                "<b>Recommended Action</b>", styles["Heading3"]
            )
        )
        story.append(
            Paragraph(
                top_recommendation.get("action", ""),
                body,
            )
        )
        if top_recommendation.get("expected_impact"):
            story.append(
                Paragraph(
                    f"Expected Impact: {top_recommendation['expected_impact']}",
                    body,
                )
            )
        if top_recommendation.get("timeline"):
            story.append(
                Paragraph(
                    f"Timeline: {top_recommendation['timeline']}",
                    body,
                )
            )

    # =====================================================
    # PAGE 3 — RISKS & WATCHOUTS (OPTIONAL)
    # =====================================================
    if risks:
        story.append(PageBreak())
        story.append(Paragraph("Risks & Watchouts", h1))
        story.append(Spacer(1, 10))

        for r in risks:
            story.append(
                Paragraph(
                    f"• {r.get('title')}: {r.get('so_what','')}",
                    body,
                )
            )
            story.append(Spacer(1, 6))

    # =====================================================
    # BUILD
    # =====================================================
    doc.build(story)
    return str(output_path)
