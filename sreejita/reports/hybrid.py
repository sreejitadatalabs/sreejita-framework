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
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.insights import correlation_insights
from sreejita.core.recommendations import generate_recommendations
from sreejita.core.schema import detect_schema
from sreejita.domains.router import apply_domain

from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist
from sreejita.visuals.categorical import bar
from sreejita.visuals.correlation import heatmap


# ------------------------------
# Header / Footer
# ------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(cm, A4[1] - 1 * cm, "Sreejita Data Labs — Hybrid Automated EDA")
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawString(
        cm,
        0.7 * cm,
        f"Confidential • Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    )
    canvas.restoreState()


# ------------------------------
# Safe Data Loader
# ------------------------------
def load_dataframe(input_path: str) -> pd.DataFrame:
    if input_path.lower().endswith(".csv"):
        try:
            return pd.read_csv(input_path)
        except UnicodeDecodeError:
            return pd.read_csv(input_path, encoding="latin1")
    return pd.read_excel(input_path)


# ------------------------------
# Main Report Runner
# ------------------------------
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    # Output handling
    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"hybrid_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    output_path = str(output_path)  # ✅ CRITICAL FIX

    # Load data
    df_raw = load_dataframe(input_path)

    # Clean
    date_col = config.get("dataset", {}).get("date")
    result = clean_dataframe(df_raw, [date_col] if date_col else None)
    df = result["df"]

    # Domain routing
    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    # Schema detection
    schema = detect_schema(df)

    # Image output
    img_dir = Path("hybrid_images")
    img_dir.mkdir(exist_ok=True)

    images = {}
    sales_col = config.get("dataset", {}).get("sales")

    if date_col and sales_col in df.columns:
        images["trend"] = img_dir / "trend.png"
        plot_monthly(df, date_col, sales_col, images["trend"])

    for col in schema["numeric"]:
        images[f"hist_{col}"] = img_dir / f"hist_{col}.png"
        hist(df, col, images[f"hist_{col}"])

    for cat in schema["categorical"]:
        images[f"bar_{cat}"] = img_dir / f"bar_{cat}.png"
        bar(df, cat, images[f"bar_{cat}"])

    images["corr"] = img_dir / "corr.png"
    heatmap(df.select_dtypes("number"), images["corr"])

    # KPIs & Insights
    kpis = compute_kpis(df, sales_col, config.get("dataset", {}).get("profit"))
    insights = correlation_insights(df, sales_col)
    recs = generate_recommendations(df)

    # Build PDF
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], alignment=1)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    story = [Paragraph("Hybrid Automated Data Report", title), Spacer(1, 12)]

    # KPI table
    rows = []
    items = list(kpis.items())
    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i + 1] if i + 1 < len(items) else ("", "")
        rows.append(
            [
                Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", styles["BodyText"]),
                Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", styles["BodyText"]),
            ]
        )

    table = Table(rows, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.extend([table, Spacer(1, 12)])

    # Insights
    story.append(Paragraph("Insights", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    # Visuals
    story.append(PageBreak())
    for img in images.values():
        if img.exists():
            story.append(Image(str(img), width=16 * cm, height=5 * cm))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    return output_path
