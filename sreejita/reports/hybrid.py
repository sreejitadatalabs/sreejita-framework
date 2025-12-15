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
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
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


# -------------------------------------------------
# Header / Footer
# -------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(
        cm, A4[1] - 1 * cm,
        "Sreejita Framework — Hybrid Decision Report"
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
    input_path = str(input_path)
    if input_path.lower().endswith(".csv"):
        try:
            return pd.read_csv(input_path)
        except UnicodeDecodeError:
            return pd.read_csv(input_path, encoding="latin1")
    return pd.read_excel(input_path)


# -------------------------------------------------
# Executive Snapshot (NEW – v1.9.1)
# -------------------------------------------------
def generate_executive_snapshot(df, schema, sales_col, profit_col):
    snapshot = {}

    snapshot["rows"] = len(df)
    snapshot["columns"] = df.shape[1]

    if sales_col and sales_col in df.columns and "region" in df.columns:
        snapshot["top_region"] = (
            df.groupby("region")[sales_col].sum().idxmax()
        )

    if sales_col and sales_col in df.columns and "segment" in df.columns:
        snapshot["top_segment"] = (
            df.groupby("segment")[sales_col].sum().idxmax()
        )

    if (
        profit_col in df.columns
        and "discount" in schema["numeric_measures"]
    ):
        corr = df[[profit_col, "discount"]].corr().iloc[0, 1]
        if corr < -0.2:
            snapshot["risk"] = (
                f"Discount negatively impacts profit (r = {corr:.2f})."
            )

    return snapshot


# -------------------------------------------------
# Main runner
# -------------------------------------------------
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    # Output path
    if output_path is None:
        output_dir = input_path.parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"Hybrid_Report_v3_1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    output_path = str(output_path)

    # Load & clean
    df_raw = load_dataframe(input_path)
    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    result = clean_dataframe(df_raw, [date_col] if date_col else None)
    df = result["df"]

    # Domain routing
    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    # Schema
    schema = detect_schema(df)
    numeric_cols = schema["numeric_measures"]
    categorical_cols = schema["categorical"]

    # KPIs
    kpis = compute_kpis(df, sales_col, profit_col)

    # Executive snapshot
    snapshot = generate_executive_snapshot(
        df, schema, sales_col, profit_col
    )

    # -------------------------------------------------
    # Visuals (reduced noise)
    # -------------------------------------------------
    img_dir = Path("hybrid_images")
    img_dir.mkdir(exist_ok=True)
    images = {}

    if date_col and sales_col in numeric_cols:
        images["trend"] = img_dir / "trend.png"
        plot_monthly(df, date_col, sales_col, images["trend"])

    for col in numeric_cols:
        if col in {sales_col, profit_col, "discount"}:
            images[f"hist_{col}"] = img_dir / f"hist_{col}.png"
            hist(df, col, images[f"hist_{col}"])

    for cat in categorical_cols:
        if cat in {"region", "segment", "ship_mode", "category"}:
            images[f"bar_{cat}"] = img_dir / f"bar_{cat}.png"
            bar(df, cat, images[f"bar_{cat}"])

    if len(numeric_cols) >= 2:
        images["corr"] = img_dir / "corr.png"
        heatmap(df[numeric_cols], images["corr"])

    # Insights & recommendations
    insights = correlation_insights(df, target=sales_col)
    recommendations = generate_recommendations(df)

    # -------------------------------------------------
    # Build PDF
    # -------------------------------------------------
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

    story = []

    # ========== PAGE 1: EXECUTIVE SNAPSHOT ==========
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))

    kpi_rows = []
    items = list(kpis.items())
    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i + 1] if i + 1 < len(items) else ("", "")
        kpi_rows.append([
            Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", styles["BodyText"]),
            Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", styles["BodyText"]),
        ])

    table = Table(kpi_rows, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    summary = (
        f"This dataset contains <b>{snapshot['rows']:,}</b> records across "
        f"<b>{snapshot['columns']}</b> variables."
    )

    if "top_region" in snapshot:
        summary += f" Sales are led by the <b>{snapshot['top_region']}</b> region."

    if "top_segment" in snapshot:
        summary += f" The <b>{snapshot['top_segment']}</b> segment drives the most revenue."

    story.append(Paragraph(summary, styles["BodyText"]))

    if "risk" in snapshot:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"• {snapshot['risk']}", styles["BodyText"]))

    story.append(PageBreak())

    # ========== INSIGHTS ==========
    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(f"• {ins}", styles["BodyText"]))

    story.append(PageBreak())

    # ========== VISUALS ==========
    for img in images.values():
        if img.exists():
            story.append(Image(str(img), width=16 * cm, height=5 * cm))

    story.append(PageBreak())

    # ========== RECOMMENDATIONS ==========
    story.append(Paragraph("Recommendations", styles["Heading2"]))
    for r in recommendations[:5]:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    return output_path
