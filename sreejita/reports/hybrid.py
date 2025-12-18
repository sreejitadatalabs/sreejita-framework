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
)

from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpi_normalizer import KPI_REGISTRY


# =====================================================
# KPI Formatting (Domain-Neutral)
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
        return f"${value:,.0f}"

    if contract.unit == "percent":
        return f"{value:.1f}%"

    if contract.unit == "count":
        return f"{int(value):,}"

    return str(value)


# =====================================================
# HYBRID REPORT (v2.x FINAL)
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    # -------------------------
    # Load & Clean
    # -------------------------
    df_raw = pd.read_csv(input_path, encoding="latin1")
    df = clean_dataframe(df_raw)["df"]

    # -------------------------
    # Domain & Policy
    # -------------------------
    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy)

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
    # PAGE 1 — EXECUTIVE BRIEF + SNAPSHOT
    # =====================================================
    story.append(Paragraph("EXECUTIVE BRIEF (1-MINUTE READ)", h1))
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

    story.append(Spacer(1, 12))
    story.append(Paragraph("Executive Snapshot", h2))
    story.append(Spacer(1, 6))

    if kpis:
        for k, v in kpis.items():
            label = k.replace("_", " ").title()
            story.append(
                Paragraph(
                    f"<b>{label}:</b> {format_kpi_value(k, v)}",
                    body,
                )
            )
    else:
        story.append(Paragraph("No KPIs derived from this dataset.", body))

    story.append(PageBreak())

    # =====================================================
    # PAGE 2–3 — VISUAL EVIDENCE (MAX 4)
    # =====================================================
    story.append(Paragraph("Visual Evidence", h1))
    story.append(Spacer(1, 10))

    if visuals:
        for idx, v in enumerate(visuals[:4], start=1):
            story.append(Image(str(v["path"]), width=14 * cm, height=8 * cm))
            if v.get("caption"):
                story.append(Paragraph(v["caption"], body))
            story.append(Spacer(1, 16))

            if idx in (2,):
                story.append(PageBreak())
    else:
        story.append(Paragraph("No visual evidence generated for this dataset.", body))

    story.append(PageBreak())

    # =====================================================
    # PAGE 4 — KEY INSIGHTS + RECOMMENDATIONS
    # =====================================================
    story.append(Paragraph("Key Insights", h1))
    story.append(Spacer(1, 10))

    for ins in insights:
        value = ins.get("value")
        title = ins["title"]

        if value:
            heading = f"[{ins['level']}] {title} — {value}"
        else:
            heading = f"[{ins['level']}] {title}"

        story.append(Paragraph(f"<b>{heading}</b>", body))
        story.append(Paragraph(ins.get("why", ""), body))
        story.append(Paragraph(ins.get("so_what", ""), body))
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Recommendations", h2))
    story.append(Spacer(1, 6))

    if recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            story.append(
                Paragraph(f"<b>{idx}. {rec.get('action','Action')}</b>", body)
            )
            if rec.get("rationale"):
                story.append(Paragraph(rec["rationale"], body))
            if rec.get("expected_impact"):
                story.append(Paragraph(f"Impact: {rec['expected_impact']}", body))
            if rec.get("timeline"):
                story.append(Paragraph(f"Timeline: {rec['timeline']}", body))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No recommendations generated.", body))

    story.append(PageBreak())

    # =====================================================
    # PAGE 5 — RISKS
    # =====================================================
    story.append(Paragraph("Risks", h1))
    story.append(Spacer(1, 10))

    risks = [i for i in insights if i.get("level") == "RISK"]

    if risks:
        for r in risks:
            story.append(Paragraph(f"<b>{r['title']}</b>", body))
            story.append(Paragraph(r.get("so_what", ""), body))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No critical risks detected.", body))

    # =====================================================
    # BUILD
    # =====================================================
    doc.build(story)
    return str(output_path)
