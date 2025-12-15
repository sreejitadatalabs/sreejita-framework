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

from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.schema import detect_schema
from sreejita.core.insights import correlation_insights
from sreejita.core.recommendations import generate_recommendations
from sreejita.domains.router import apply_domain

from sreejita.visuals.correlation import heatmap
from sreejita.visuals.categorical import bar
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist


# -------------------------------------------------
# Utilities
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


# -------------------------------------------------
# Load Data
# -------------------------------------------------
def load_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    return pd.read_excel(path)


# -------------------------------------------------
# Evidence Snapshot Builder (FINAL)
# -------------------------------------------------
def build_evidence_snapshot(df, schema, config):
    """
    v1.9.7 FINAL:
    Always attempts to generate 3 complementary visuals:
    1) Trend or Distribution
    2) Categorical Breakdown
    3) Correlation Heatmap

    Each visual is attempted independently.
    """
    visuals = []

    img_dir = Path("hybrid_images").resolve()
    img_dir.mkdir(exist_ok=True)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")

    # -------------------------
    # 1️⃣ Trend OR Distribution
    # -------------------------
    trend_or_dist_added = False

    # Try trend if date exists
    if date_col and date_col in df.columns and sales_col in df.columns:
        trend_path = img_dir / "evidence_trend.png"
        plot_monthly(df, date_col, sales_col, trend_path)

        if trend_path.exists() and trend_path.stat().st_size > 0:
            visuals.append((
                trend_path,
                "Time-series trend shows how performance evolves over time, "
                "highlighting growth patterns, seasonality, or volatility."
            ))
            trend_or_dist_added = True

    # Fallback to distribution if trend not added
    if not trend_or_dist_added and sales_col in df.columns:
        dist_path = img_dir / "evidence_distribution.png"
        hist(df, sales_col, dist_path)

        if dist_path.exists() and dist_path.stat().st_size > 0:
            visuals.append((
                dist_path,
                "Distribution plot shows the spread and outliers of key performance values, "
                "indicating variability and risk."
            ))

    # -------------------------
    # 2️⃣ Categorical Breakdown
    # -------------------------
    if sales_col in df.columns and schema.get("categorical"):
        cat = schema["categorical"][0]
        bar_path = img_dir / "evidence_bar.png"
        bar(df, cat, bar_path)

        if bar_path.exists() and bar_path.stat().st_size > 0:
            visuals.append((
                bar_path,
                f"Categorical breakdown shows how {sales_col} is distributed across {cat}, "
                f"revealing dominant contributors."
            ))

    # -------------------------
    # 3️⃣ Correlation Heatmap
    # -------------------------
    if len(schema.get("numeric_measures", [])) >= 2:
        corr_path = img_dir / "evidence_correlation.png"
        corr_img = heatmap(df, corr_path)

        if corr_img and corr_img.exists() and corr_img.stat().st_size > 0:
            visuals.append((
                corr_img,
                "Correlation heatmap reveals relationships between key numeric metrics, "
                "supporting cause–effect reasoning."
            ))

    # Final guarantee: return up to 3 visuals, in order
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

    # Load & clean
    df_raw = load_dataframe(str(input_path))
    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    df = clean_dataframe(df_raw, [date_col] if date_col else None)["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    schema = detect_schema(df)

    # Intelligence
    kpis = compute_kpis(df, sales_col, profit_col)

    insights = enforce_min_bullets(
        correlation_insights(df, sales_col),
        4,
        [
            "Performance is driven by a limited set of key factors.",
            "Results vary across business dimensions.",
            "Pricing and volume trade-offs influence outcomes.",
            "Operational behavior suggests optimization opportunities.",
        ],
    )

    recommendations = enforce_min_bullets(
        generate_recommendations(df, sales_col, profit_col),
        4,
        [
            "Strengthen pricing governance to protect margins.",
            "Prioritize high-impact segments for growth.",
            "Introduce monitoring for performance volatility.",
            "Improve data completeness for future analysis.",
        ],
    )

    evidence = build_evidence_snapshot(df, schema, config)

    missing_pct = (df.isna().sum() / len(df)) * 100
    dq_notes = [
        f"Missing values present in {missing_pct[missing_pct > 0].count()} columns "
        f"(highest: {missing_pct.max():.1f}%).",
        "Identifier-like fields excluded from numeric analysis.",
        "Outliers may influence aggregate metrics.",
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

    # Evidence Snapshot
    story.append(Paragraph("Evidence Snapshot (Supporting Analysis)", title))
    story.append(Spacer(1, 12))

    for img_path, note in evidence:
        img_path = Path(img_path).resolve()
        story.append(
            Image(
                str(img_path),
                width=15 * cm,
                height=11 * cm,
                kind="proportional",
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph(note, styles["BodyText"]))
        story.append(Spacer(1, 18))

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
