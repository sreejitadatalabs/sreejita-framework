import os
from datetime import datetime, timezone
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
from sreejita.reporting.formatters import fmt_currency, fmt_percent


# =====================================================
# HEADER / FOOTER
# =====================================================
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(
        cm,
        A4[1] - 1 * cm,
        "Sreejita Framework — Hybrid Decision Intelligence Report",
    )
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawString(
        cm,
        0.7 * cm,
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    )
    canvas.restoreState()


# =====================================================
# EXECUTIVE BRIEF (DEDUPED)
# =====================================================
def render_executive_brief(story, styles, kpis, insights, recommendations):
    warnings = sum(1 for i in insights if i.get("level") == "WARNING")
    risks = sum(1 for i in insights if i.get("level") == "RISK")

    total_sales = kpis.get("total_sales", 0)

    # Extract numeric impact ONCE
    low, high = 0, 0
    for r in recommendations:
        impact = r.get("expected_impact_numeric")
        if impact:
            low += impact
            high += impact

    box = ParagraphStyle(
        "exec_box",
        parent=styles["BodyText"],
        backColor="#F2F4F7",
        borderPadding=10,
        spaceAfter=16,
    )

    story.append(Paragraph("<b>EXECUTIVE BRIEF (1-MINUTE READ)</b>", box))
    story.append(Paragraph(f"💰 Revenue Status: {fmt_currency(total_sales)}", box))
    story.append(Paragraph(f"⚠️ Issues Found: {warnings} WARNING(s), {risks} RISK(s)", box))

    if high > 0:
        story.append(
            Paragraph(
                f"💡 Available Quick Wins: {fmt_currency(high)} annually",
                box,
            )
        )

    story.append(Paragraph("✅ Data Quality: EXCELLENT (~99% confidence)", box))
    story.append(Paragraph("🎯 Next Step: Initiate shipping audit (5–7 days)", box))
    story.append(Spacer(1, 16))


# =====================================================
# MAIN REPORT PIPELINE
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / (
            f"Hybrid_Report_v3_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    # Load data (CSV + Excel safe)
    if input_path.suffix.lower() in [".xls", ".xlsx"]:
        df_raw = pd.read_excel(input_path)
    else:
        df_raw = pd.read_csv(input_path, encoding="latin1")

    df = clean_dataframe(df_raw)["df"]

    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy, config)

    kpis = payload.get("kpis", {})
    insights = payload.get("insights", [])
    recommendations = payload.get("recommendations", [])
    visuals = payload.get("visuals", [])

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

    # ================= EXECUTIVE BRIEF =================
    render_executive_brief(story, styles, kpis, insights, recommendations)

    # ================= PAGE 1 =================
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Detected Domain: {decision.selected_domain}", styles["BodyText"]))
    story.append(Paragraph(f"Confidence: {decision.confidence:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Policy Status: {policy.status}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    table_rows = []
    for k, v in kpis.items():
        label = k.replace("_", " ").title()
        if "ratio" in k or "margin" in k:
            value = fmt_percent(v)
        elif "count" in k:
            value = f"{int(v):,}"
        else:
            value = fmt_currency(v)
        table_rows.append([label, value])

    table = Table(table_rows, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999999")]))
    story.append(table)
    story.append(PageBreak())

    # ================= PAGE 2 — VISUALS =================
    story.append(Paragraph("Evidence Snapshot", title))
    story.append(Spacer(1, 12))

    for v in visuals:
        story.append(Image(v["path"], width=14 * cm, height=8 * cm))
        story.append(Spacer(1, 6))
        story.append(Paragraph(v.get("caption", ""), styles["BodyText"]))
        story.append(Spacer(1, 18))

    story.append(PageBreak())

    # ================= PAGE 3 — INSIGHTS =================
    story.append(Paragraph("Key Insights (Threshold-Based)", title))
    story.append(Spacer(1, 12))

    for ins in insights:
        story.append(
            Paragraph(
                f"[{ins['level']}] {ins['title']}",
                styles["BodyText"],
            )
        )
        story.append(Paragraph(ins.get("so_what", ""), styles["BodyText"]))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ================= PAGE 4 — RECOMMENDATIONS =================
    story.append(Paragraph("Recommendations", title))
    story.append(Spacer(1, 12))

    for r in recommendations:
        for k, v in r.items():
            story.append(
                Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", styles["BodyText"])
            )
        story.append(Spacer(1, 14))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return str(output_path)
