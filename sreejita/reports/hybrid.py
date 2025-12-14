import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from typing import Optional
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


def run(input_path: str, config: dict, output_path: Optional[str] = None):
    if output_path is None:
        output_path = f"hybrid_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    # Load data
    def load_dataframe(input_path: str):
        if input_path.endswith(".csv"):
            try:
                return pd.read_csv(input_path)
            except UnicodeDecodeError:
                return pd.read_csv(input_path, encoding="latin1")
        else:
            return pd.read_excel(input_path)


    # Clean
    date_col = config["dataset"].get("date")
    result = clean_dataframe(df_raw, [date_col] if date_col else None)
    df = result["df"]

    # Domain
    df = apply_domain(df, config["domain"]["name"])

    # Schema
    schema = detect_schema(df)

    img_dir = Path("hybrid_images")
    img_dir.mkdir(exist_ok=True)

    images = {}

    sales_col = config["dataset"].get("sales")

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

    kpis = compute_kpis(df, sales_col, config["dataset"].get("profit"))
    insights = correlation_insights(df, sales_col)
    recs = generate_recommendations(df)

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], alignment=1)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2.5 * cm, bottomMargin=2 * cm)

    story = [Paragraph("Hybrid Automated Data Report", title), Spacer(1, 12)]

    rows = []
    items = list(kpis.items())
    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i + 1] if i + 1 < len(items) else ("", "")
        rows.append([
            Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", styles["BodyText"]),
            Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", styles["BodyText"]),
        ])

    table = Table(rows, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.extend([table, Spacer(1, 12)])

    story.append(Paragraph("Insights", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    story.append(PageBreak())
    for img in images.values():
        if img.exists():
            story.append(Image(str(img), width=16 * cm, height=5 * cm))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
