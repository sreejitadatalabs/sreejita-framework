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
    PageBreak,
    Image,
    Table,
    TableStyle,
)

from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.reporting.generic_visuals import generate_generic_visuals
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpi_normalizer import KPI_REGISTRY


# =====================================================
# KPI Formatting
# =====================================================
def format_kpi_value(kpi, value):
    contract = KPI_REGISTRY.get(kpi)
    if value is None:
        return "N/A"
    if not contract:
        return str(value)
    if contract.unit == "currency":
        return f"${value:,.2f}"
    if contract.unit == "percent":
        return f"{value:.1f}%"
    return str(value)


# =====================================================
# HYBRID REPORT (FINAL v2.x)
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    df = clean_dataframe(pd.read_csv(input_path, encoding="latin1"))["df"]

    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy)

    kpis = payload.get("kpis", {})
    insights = payload.get("insights", [])
    recommendations = payload.get("recommendations", [])
    visuals = payload.get("visuals", [])

    if not visuals:
        visuals = generate_generic_visuals(df, input_path.parent / "reports" / "visuals")

    styles = getSampleStyleSheet()
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

    # ================= PAGE 1 — EXECUTIVE BRIEF =================
    warnings = sum(1 for i in insights if i["level"] == "WARNING")
    risks = sum(1 for i in insights if i["level"] == "RISK")

    story.append(Paragraph("EXECUTIVE BRIEF (1-MINUTE READ)", h1))
    story.append(Paragraph(f"Detected Domain: {decision.selected_domain}", body))
    story.append(Paragraph(f"Issues Found: {warnings} WARNING(s), {risks} RISK(s)", body))

    if decision.selected_domain == "retail" and "total_sales" in kpis:
        story.append(Paragraph(f"Revenue Status: ${kpis['total_sales']:,.0f}", body))

    if recommendations:
        story.append(Paragraph(f"Next Step: {recommendations[0].get('action')}", body))

    story.append(Spacer(1, 12))

    # ================= EXECUTIVE SNAPSHOT (TABLE) =================
    rows = [["Metric", "Value"]]
    for k, v in kpis.items():
        rows.append([k.replace("_", " ").title(), format_kpi_value(k, v)])

    if len(rows) == 1:
        rows.append(["KPIs", "Not available for this dataset"])

    table = Table(rows, colWidths=[8 * cm, 6 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999999")]))
    story.append(table)

    story.append(PageBreak())

    # ================= PAGE 2–3 — VISUALS =================
    story.append(Paragraph("Visual Evidence", h1))
    for idx, v in enumerate(visuals[:4]):
        story.append(Image(str(v["path"]), width=14 * cm, height=8 * cm))
        story.append(Paragraph(v.get("caption", ""), body))
        story.append(Spacer(1, 16))
        if idx == 1:
            story.append(PageBreak())

    story.append(PageBreak())

    # ================= PAGE 4 — INSIGHTS + RECOMMENDATIONS =================
    story.append(Paragraph("Key Insights", h1))
    for i in insights:
        story.append(Paragraph(f"[{i['level']}] {i['title']}", body))
        story.append(Paragraph(i.get("so_what", ""), body))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Recommendations", h1))
    for r in recommendations:
        story.append(Paragraph(f"- {r.get('action')}", body))

    story.append(PageBreak())

    # ================= PAGE 5 — RISKS =================
    story.append(Paragraph("Risks", h1))
    risks_only = [i for i in insights if i["level"] == "RISK"]

    if risks_only:
        for r in risks_only:
            story.append(Paragraph(r["title"], body))
            story.append(Paragraph(r.get("so_what", ""), body))
    else:
        story.append(Paragraph("No critical risks detected.", body))

    doc.build(story)
    return str(output_path)
