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

# -------------------------
# v2.x orchestration
# -------------------------
from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.domains.router import decide_domain, apply_domain
from sreejita.policy.engine import PolicyEngine

# -------------------------
# v1 core (fallback)
# -------------------------
from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.schema import detect_schema
from sreejita.core.insights import correlation_insights
from sreejita.core.recommendations import generate_recommendations

# -------------------------
# v1 visuals (fallback)
# -------------------------
from sreejita.visuals.correlation import heatmap
from sreejita.visuals.categorical import bar
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist


# =====================================================
# SAFETY
# =====================================================
def safe(val, fallback):
    return val if val not in [None, "", "nan"] else fallback


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
# DATA LOADER
# =====================================================
def load_dataframe(path: Path):
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    elif path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type")


# =====================================================
# MAIN PIPELINE — v2.7
# =====================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_v3_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    # -------------------------
    # Load & clean data
    # -------------------------
    df_raw = load_dataframe(input_path)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    df = clean_dataframe(df_raw, [date_col] if date_col else None)["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    # -------------------------
    # Decision + policy
    # -------------------------
    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    payload = generate_report_payload(df, decision, policy)

    if payload:
        kpis = payload["kpis"]
        insights = payload["insights"]
        recs = payload["recommendations"]
        visuals = [(v["path"], v["caption"]) for v in payload.get("visuals", [])]
    else:
        kpis = compute_kpis(df, sales_col, profit_col)
        insights = [{"title": t, "level": "INFO"} for t in correlation_insights(df, sales_col)]
        recs = generate_recommendations(df, sales_col, profit_col)
        visuals = []

    # -------------------------
    # PDF setup
    # -------------------------
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

    # ================= PAGE 1 =================
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Detected Domain: {decision.selected_domain}", styles["BodyText"]))
    story.append(Paragraph(f"Confidence: {decision.confidence:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Policy Status: {policy.status}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    kpi_table = Table(list(kpis.items()), colWidths=[7 * cm, 7 * cm])
    kpi_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999999")]))
    story.append(kpi_table)
    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph("Evidence Snapshot", title))
    story.append(Spacer(1, 12))

    for img, cap in visuals:
        story.append(Image(str(img), width=14 * cm, height=8 * cm))
        story.append(Spacer(1, 6))
        story.append(Paragraph(cap, styles["BodyText"]))
        story.append(Spacer(1, 18))

    story.append(PageBreak())

    # ================= PAGE 3 =================
    story.append(Paragraph("Key Insights (Threshold-Based)", title))
    story.append(Spacer(1, 12))

    if insights and isinstance(insights, list):

        for ins in insights:
            level = ins.get("level", "INFO")
            title_txt = ins.get("title", "Insight")
            value = ins.get("value", "")
            what = ins.get("what", "")
            why = ins.get("why", "")
            so_what = ins.get("so_what", "")

            icon = "🔴" if level == "RISK" else "⚠️" if level == "WARNING" else "🟢"

            story.append(
                Paragraph(
                    f"<b>{icon} [{level}] {title_txt}</b>"
                    + (f" — {value}" if value != "" else ""),
                    styles["BodyText"],
                )
            )

            if what:
                story.append(Paragraph(f"<i>What:</i> {what}", styles["BodyText"]))
            if why:
                story.append(Paragraph(f"<i>Why:</i> {why}", styles["BodyText"]))
            if so_what:
                story.append(Paragraph(f"<i>So what:</i> {so_what}", styles["BodyText"]))

            story.append(Spacer(1, 14))
    else:
        story.append(
            Paragraph(
                "No material operational risks detected based on defined thresholds.",
                styles["BodyText"],
            )
        )

    story.append(PageBreak())

    # ================= PAGE 4 =================
    story.append(Paragraph("Recommendations", title))
    story.append(Spacer(1, 12))

    for r in recs:
        story.append(Paragraph(f"<b>Action:</b> {r.get('action','')}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Priority:</b> {r.get('priority','')}", styles["BodyText"]))
        story.append(
            Paragraph(
                f"<b>Expected Impact:</b> {r.get('expected_impact','')}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 12))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return str(output_path)
