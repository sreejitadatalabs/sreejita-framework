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
# KPI Formatting (Contract-Driven, No Guessing)
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
# HYBRID REPORT (DEEP DIVE)
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
    # Domain Decision & Policy
    # -------------------------
    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    # -------------------------
    # Analysis Payload
    # -------------------------
    payload = generate_report_payload(df, decision, policy)
    if payload is None:
        raise RuntimeError("Report payload generation failed")

    kpis = payload.get("kpis", {})
    insights = payload.get("insights", [])
    recommendations = payload.get("recommendations", [])
    visuals = payload.get("visuals", [])
    narrative = payload.get("narrative", {})

    # -------------------------
    # PDF Setup
    # -------------------------
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("h1", parent=styles["Heading1"])
    h2 = ParagraphStyle("h2", parent=styles["Heading2"])
    body = styles["BodyText"]
    italic = styles["Italic"]

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
    # 1️⃣ OVERVIEW
    # =====================================================
    story.append(Paragraph("Overview", h1))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"<b>Detected Domain:</b> {decision.selected_domain}<br/>"
            f"<b>Confidence:</b> {decision.confidence:.2f}<br/>"
            f"<b>Policy Status:</b> {policy.status}",
            body,
        )
    )

    if narrative.get("overview"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(narrative["overview"], body))

    story.append(PageBreak())

    # =====================================================
    # 2️⃣ KPIs
    # =====================================================
    story.append(Paragraph("Key Performance Indicators", h1))
    story.append(Spacer(1, 10))

kpi_rows = [["Metric", "Value"]]
    kpi_rows.extend([
        [k.replace("_", " ").title(), format_kpi_value(k, v)]
        for k, v in kpis.items()
    ])
    kpi_table = Table(kpi_rows, colWidths=[9 * cm, 5 * cm])
    kpi_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
            ("BACKGROUND", (0, 0), (-1, 0), "#F2F4F7"),
        ])
    )
    story.append(kpi_table)

    story.append(PageBreak())

    # =====================================================
    # 3️⃣ VISUAL EVIDENCE
    # =====================================================
    story.append(Paragraph("Visual Evidence", h1))
    story.append(Spacer(1, 10))

    if visuals:
        for v in visuals:
            story.append(Image(str(v["path"]), width=14 * cm, height=8 * cm))
            if v.get("caption"):
                story.append(Paragraph(v["caption"], body))
            story.append(Spacer(1, 14))
    else:
        story.append(Paragraph("No visuals generated for this dataset.", body))

    story.append(PageBreak())

    # =====================================================
    # 4️⃣ INSIGHTS
    # =====================================================
    story.append(Paragraph("Key Insights", h1))
    story.append(Spacer(1, 10))

    for idx, ins in enumerate(insights, start=1):
        story.append(
            Paragraph(
                f"<b>{idx}. [{ins['level']}] {ins['title']}</b>",
                body,
            )
        )
        story.append(Paragraph(ins.get("why", ""), body))
        story.append(Paragraph(ins.get("so_what", ""), body))

        if "semantic_warning" in ins:
            story.append(
                Paragraph(
                    f"⚠ {ins['semantic_warning']}",
                    italic,
                )
            )

        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # =====================================================
    # 5️⃣ RECOMMENDATIONS
    # =====================================================
    story.append(Paragraph("Recommendations", h1))
    story.append(Spacer(1, 10))

    if recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            story.append(
                Paragraph(
                    f"<b>{idx}. {rec.get('action','Action')}</b>",
                    body,
                )
            )

            if rec.get("rationale"):
                story.append(
                    Paragraph(
                        f"<b>Rationale:</b> {rec['rationale']}",
                        body,
                    )
                )

            if rec.get("expected_impact"):
                story.append(
                    Paragraph(
                        f"<b>Expected Impact:</b> {rec['expected_impact']}",
                        body,
                    )
                )

            if rec.get("timeline"):
                story.append(
                    Paragraph(
                        f"<b>Timeline:</b> {rec['timeline']}",
                        body,
                    )
                )

            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No recommendations generated.", body))

    # =====================================================
    # BUILD
    # =====================================================
    doc.build(story)
    return str(output_path)
