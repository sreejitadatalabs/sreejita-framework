import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak, Image
)

from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.schema import detect_schema
from sreejita.core.insights import correlation_insights
from sreejita.core.recommendations import generate_recommendations
from sreejita.domains.router import apply_domain

from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.categorical import bar
from sreejita.visuals.correlation import heatmap


# -------------------------------------------------
# Utilities (Safety + Narrative)
# -------------------------------------------------
def safe_label(value, fallback):
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
def load_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    return pd.read_excel(path)


# -------------------------------------------------
# Executive Snapshot
# -------------------------------------------------
def build_executive_snapshot(df, kpis, sales_col):
    snapshot = {"kpis": list(kpis.items())[:6]}

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
# Evidence Snapshot Builder (v1.9.7)
# -------------------------------------------------
def build_evidence_snapshot(df, schema, config):
    visuals = []

    img_dir = Path("hybrid_images")
    img_dir.mkdir(exist_ok=True)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")

    # 1️⃣ Primary Trend
    if date_col in df.columns and sales_col in df.columns:
        path = img_dir / "evidence_trend.png"
        plot_monthly(df, date_col, sales_col, path)
        visuals.append((
            path,
            "The time trend highlights overall performance movement, "
            "helping identify growth patterns, seasonality, or instability."
        ))

    # 2️⃣ Primary Breakdown
    if schema["categorical"] and sales_col in df.columns:
        cat = schema["categorical"][0]
        path = img_dir / "evidence_breakdown.png"
        bar(df, cat, path)
        visuals.append((
            path,
            f"The categorical breakdown shows how performance is distributed across {cat}, "
            f"indicating concentration or diversification."
        ))

    # 3️⃣ Primary Relationship
    if len(schema["numeric_measures"]) >= 2:
        path = img_dir / "evidence_correlation.png"
        heatmap(df[schema["numeric_measures"]], path)
        visuals.append((
            path,
            "The correlation view reveals relationships between key metrics, "
            "supporting cause–effect reasoning behind insights."
        ))

    return visuals[:3]


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

    df_raw = load_dataframe(str(input_path))

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    df = clean_dataframe(df_raw, [date_col] if date_col else None)["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    schema = detect_schema(df)

    # Core intelligence
    kpis = compute_kpis(df, sales_col, profit_col)
    snapshot = build_executive_snapshot(df, kpis, sales_col)

    insights = enforce_min_bullets(
        correlation_insights(df, sales_col),
        4,
        [
            "Performance is driven by a limited set of key factors.",
            "Results vary significantly across business dimensions.",
            "Pricing and volume trade-offs influence outcomes.",
            "Operational behavior suggests optimization opportunities."
        ]
    )

    recommendations = enforce_min_bullets(
        generate_recommendations(df, sales_col, profit_col),
        4,
        [
            "Strengthen pricing governance to protect margins.",
            "Prioritize high-impact segments for growth.",
            "Introduce monitoring for performance volatility.",
            "Improve data completeness for future analysis."
        ]
    )

    # Evidence snapshot
    evidence = build_evidence_snapshot(df, schema, config)

    # Data quality
    missing_pct = (df.isna().sum() / len(df)) * 100
    dq_notes = [
        f"Missing values present in {missing_pct[missing_pct > 0].count()} columns "
        f"(highest: {missing_pct.max():.1f}%).",
        "Identifier-like fields excluded from numeric analysis.",
        "Outliers may influence aggregate metrics."
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
        f"Performance is driven primarily by "
        f"<b>{safe_label(snapshot.get('top_segment'), 'key segments')}</b> "
        f"and activity across <b>{safe_label(snapshot.get('top_region'), 'multiple regions')}</b>.",
        styles["BodyText"]
    ))

    story.append(PageBreak())

    # Evidence Snapshot
    story.append(Paragraph("Evidence Snapshot (Supporting Analysis)", styles["Heading2"]))
    for img_path, note in evidence:
        if img_path.exists():
            story.append(Image(str(img_path), width=16 * cm, height=6 * cm))
            story.append(Spacer(1, 4))
            story.append(Paragraph(note, styles["BodyText"]))
            story.append(Spacer(1, 12))

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
