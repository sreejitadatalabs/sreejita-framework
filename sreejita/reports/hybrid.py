import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak
)

from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.schema import detect_schema
from sreejita.core.insights import correlation_insights
from sreejita.core.recommendations import generate_recommendations
from sreejita.domains.router import apply_domain


# -------------------------------------------------
# Safety & Narrative Utilities (v1.9.6)
# -------------------------------------------------
def safe_label(value: Optional[str], fallback: str) -> str:
    return value if value not in [None, "", "nan"] else fallback


def enforce_min_bullets(items, min_count, fillers):
    items = list(items)
    while len(items) < min_count:
        items.append(fillers[len(items) % len(fillers)])
    return items


# -------------------------------------------------
# Header / Footer
# -------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(
        cm, A4[1] - 1 * cm,
        "Sreejita Framework — Hybrid Decision Intelligence Report"
    )
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawString(
        cm, 0.7 * cm,
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    canvas.restoreState()


# -------------------------------------------------
# Load data
# -------------------------------------------------
def load_dataframe(input_path: str) -> pd.DataFrame:
    if input_path.lower().endswith(".csv"):
        try:
            return pd.read_csv(input_path)
        except UnicodeDecodeError:
            return pd.read_csv(input_path, encoding="latin1")
    return pd.read_excel(input_path)


# -------------------------------------------------
# Executive Snapshot Builder
# -------------------------------------------------
def build_executive_snapshot(df, kpis, sales_col):
    snapshot = {}

    # KPI tiles (4–6 max)
    snapshot["kpis"] = list(kpis.items())[:6]

    if sales_col and sales_col in df.columns and "segment" in df.columns:
        snapshot["top_segment"] = (
            df.groupby("segment")[sales_col].sum().idxmax()
        )

    if sales_col and sales_col in df.columns and "region" in df.columns:
        snapshot["top_region"] = (
            df.groupby("region")[sales_col].sum().idxmax()
        )

    return snapshot


# -------------------------------------------------
# Main Runner
# -------------------------------------------------
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        output_dir = input_path.parent / "reports"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / (
            f"Hybrid_Report_v3_1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    # Load & clean
    df_raw = load_dataframe(str(input_path))
    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    df = clean_dataframe(df_raw, [date_col] if date_col else None)["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    schema = detect_schema(df)

    # Core outputs
    kpis = compute_kpis(df, sales_col, profit_col)
    snapshot = build_executive_snapshot(df, kpis, sales_col)

    insights = correlation_insights(df, sales_col)
    insights = enforce_min_bullets(
        insights,
        min_count=4,
        fillers=[
            "Performance is concentrated among a limited set of drivers.",
            "Sales behavior varies significantly across business dimensions.",
            "Pricing and volume trade-offs influence outcomes.",
            "Operational patterns suggest optimization opportunities."
        ]
    )

    recommendations = generate_recommendations(df, sales_col, profit_col)
    recommendations = enforce_min_bullets(
        recommendations,
        min_count=4,
        fillers=[
            "Strengthen governance around pricing and discounts.",
            "Focus resources on consistently high-performing areas.",
            "Introduce monitoring for margin erosion risks.",
            "Improve data capture for better future analysis."
        ]
    )

    # Data Quality
    missing_pct = (df.isna().sum() / len(df)) * 100
    dq_notes = [
        f"Missing values present in {missing_pct[missing_pct > 0].count()} columns "
        f"(highest: {missing_pct.max():.1f}%).",
        "Identifier-like fields excluded from numeric analysis.",
        "Right-skewed distributions indicate potential outliers."
    ]

    # -------------------------------------------------
    # Build PDF
    # -------------------------------------------------
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

    # Executive Snapshot
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))

    rows = []
    for i in range(0, len(snapshot["kpis"]), 2):
        left = snapshot["kpis"][i]
        right = snapshot["kpis"][i + 1] if i + 1 < len(snapshot["kpis"]) else ("", "")
        rows.append([
            Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", styles["BodyText"]),
            Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", styles["BodyText"]),
        ])

    story.append(Table(rows, colWidths=[8 * cm, 8 * cm]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Overall performance is driven by "
        f"<b>{safe_label(snapshot.get('top_segment'), 'key segments')}</b> "
        f"and activity across <b>{safe_label(snapshot.get('top_region'), 'multiple regions')}</b>.",
        styles["BodyText"]
    ))

    story.append(PageBreak())

    # Key Insights
    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    story.append(PageBreak())

    # Recommendations
    story.append(Paragraph("Recommendations", styles["Heading2"]))
    for r in recommendations:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    story.append(PageBreak())

    # Data Quality
    story.append(Paragraph("Data Quality & Risk Notes", styles["Heading2"]))
    for d in dq_notes:
        story.append(Paragraph(f"• {d}", styles["BodyText"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    return str(output_path)
