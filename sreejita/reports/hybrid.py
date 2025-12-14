import os
from datetime import datetime, timezone

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
# PDF header & footer
# ------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(cm, A4[1] - 1 * cm, "Sreejita Data Labs — Hybrid Automated EDA")
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawString(
        cm,
        0.7 * cm,
        f"Confidential • Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}:
    )
    canvas.restoreState()


# ------------------------------
# Main Hybrid Runner
# ------------------------------
def run_hybrid(input_path: str, output_path: str, config: dict):
    print("[SREEJITA] Running HYBRID report...")

    # 1. Load dataset
    if input_path.endswith(".csv"):
        df_raw = pd.read_csv(input_path, low_memory=False)
    else:
        df_raw = pd.read_excel(input_path)

    # 2. Clean data
    date_col = config["dataset"].get("date")
    result = clean_dataframe(df_raw, [date_col] if date_col else None)
    df = result["df"]

        # Domain routing
    df = apply_domain(df, config["domain"]["name"])
    
    # Schema detection & column configuration
    schema = detect_schema(df)
    numeric_cols = config["analysis"].get("numeric") or schema["numeric"]
    categorical_cols = config["analysis"].get("categorical") or schema["categorical"]
    summary = result["summary"]

    # 3. Prepare image directory
    img_dir = "hybrid_images"
    os.makedirs(img_dir, exist_ok=True)

    images = {}

    # 4. Generate visuals
    sales_col = config["dataset"].get("sales", "sales")

    if date_col and sales_col in df.columns:
        images["trend"] = os.path.join(img_dir, "monthly_trend.png")
        plot_monthly(df, date_col, sales_col, images["trend"])

    for col in config["analysis"].get("numeric_distribution", []):
        if col in df.columns:
            images[f"hist_{col}"] = os.path.join(img_dir, f"hist_{col}.png")
            hist(df, col, images[f"hist_{col}"])

    for cat in config["analysis"].get("categorical", []):
        if cat in df.columns:
            images[f"bar_{cat}"] = os.path.join(img_dir, f"bar_{cat}.png")
            bar(df, cat, images[f"bar_{cat}"])

    images["corr"] = os.path.join(img_dir, "correlation.png")
    heatmap(df.select_dtypes("number"), images["corr"])

    # 5. Compute KPIs & insights
    kpis = compute_kpis(df, sales_col=sales_col, profit_col=config["dataset"].get("profit"))
    insights = correlation_insights(df, target=sales_col)
    recommendations = generate_recommendations(df)

    # 6. Build PDF
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        alignment=1,
        textColor=colors.HexColor("#0B1C3D"),
    )

    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0B1C3D"),
    )

    normal = styles["BodyText"]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # ---- Title ----
    story.append(Paragraph("Hybrid Automated Data Report", title))
    story.append(Spacer(1, 12))

    # ---- KPI Table ----
    rows = []
    items = list(kpis.items())
    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i + 1] if i + 1 < len(items) else ("", "")
        rows.append(
            [
                Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", normal),
                Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", normal),
            ]
        )

    table = Table(rows, colWidths=[8 * cm, 8 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # ---- Insights ----
    story.append(Paragraph("Automated Insights", h2))
    for ins in insights:
        story.append(Paragraph(f"• {ins}", normal))
    story.append(Spacer(1, 12))

    # ---- Recommendations ----
    story.append(Paragraph("Recommendations", h2))
    for r in recommendations:
        story.append(Paragraph(f"• {r}", normal))

    # ---- Visuals ----
    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", h2))

    for key, path in images.items():
        if os.path.exists(path):
            story.append(Image(path, width=16 * cm, height=5 * cm))
            story.append(Spacer(1, 12))

    # ---- Build ----
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    print(f"[SREEJITA] Hybrid report generated → {output_path}")
