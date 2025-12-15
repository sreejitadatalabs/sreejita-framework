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
    if input_path.lower().endswith(".csv"):
        try:
            return pd.read_csv(input_path)
        except UnicodeDecodeError:
            return pd.read_csv(input_path, encoding="latin1")
    return pd.read_excel(input_path)


# -------------------------------------------------
# Executive Snapshot
# -------------------------------------------------
def generate_executive_snapshot(df, schema, kpis, sales_col, profit_col):
    snapshot = {"kpis": kpis}

    if sales_col and sales_col in df.columns and "segment" in df.columns:
        seg_share = (
            df.groupby("segment")[sales_col].sum()
            / df[sales_col].sum()
        )
        snapshot["top_segment"] = seg_share.idxmax()
        snapshot["top_segment_share"] = seg_share.max()

    if sales_col and sales_col in df.columns and "region" in df.columns:
        snapshot["top_region"] = (
            df.groupby("region")[sales_col].sum().idxmax()
        )

    if (
        profit_col
        and profit_col in df.columns
        and "discount" in schema["numeric_measures"]
    ):
        corr = df[[profit_col, "discount"]].corr().iloc[0, 1]
        if corr < -0.2:
            snapshot["risk"] = (
                f"Discounting negatively impacts profit (r = {corr:.2f}), "
                f"indicating margin leakage risk."
            )

    return snapshot


# -------------------------------------------------
# Written Insights
# -------------------------------------------------
def generate_written_insights(df, schema, sales_col, profit_col):
    insights = []

    if sales_col and sales_col in df.columns and "segment" in df.columns:
        seg_share = (
            df.groupby("segment")[sales_col].sum()
            / df[sales_col].sum()
        )
        insights.append(
            f"{seg_share.idxmax()} segment contributes "
            f"{seg_share.max()*100:.1f}% of total sales, making it the primary growth driver."
        )

    if sales_col and sales_col in df.columns and "region" in df.columns:
        insights.append(
            f"{df.groupby('region')[sales_col].sum().idxmax()} region leads in total sales."
        )

    if (
        profit_col
        and profit_col in df.columns
        and "discount" in schema["numeric_measures"]
    ):
        corr = df[[profit_col, "discount"]].corr().iloc[0, 1]
        if corr < -0.2:
            insights.append(
                f"Discount shows a negative correlation with profit (r = {corr:.2f}), "
                f"indicating margin erosion."
            )

    return insights


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

    result = clean_dataframe(df_raw, [date_col] if date_col else None)
    df = result["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    schema = detect_schema(df)
    numeric_cols = schema["numeric_measures"]
    categorical_cols = schema["categorical"]

    # KPIs
    kpis = compute_kpis(df, sales_col, profit_col)

    snapshot = generate_executive_snapshot(
        df, schema, kpis, sales_col, profit_col
    )
    insights = generate_written_insights(
        df, schema, sales_col, profit_col
    )

    # -------------------------------------------------
    # Visuals
    # -------------------------------------------------
    img_dir = Path("hybrid_images")
    img_dir.mkdir(exist_ok=True)
    images = {}

    if date_col and sales_col and sales_col in numeric_cols:
        images["trend"] = img_dir / "trend.png"
        plot_monthly(df, date_col, sales_col, images["trend"])

    for col in numeric_cols:
        if col in {sales_col, profit_col, "discount"}:
            images[f"hist_{col}"] = img_dir / f"hist_{col}.png"
            hist(df, col, images[f"hist_{col}"])

    for cat in categorical_cols:
        if cat in {"region", "segment", "category", "ship_mode"}:
            images[f"bar_{cat}"] = img_dir / f"bar_{cat}.png"
            bar(df, cat, images[f"bar_{cat}"])

    if len(numeric_cols) >= 2:
        images["corr"] = img_dir / "corr.png"
        heatmap(df[numeric_cols], images["corr"])

    # -------------------------------------------------
    # Correlation Interpretation
    # -------------------------------------------------
    corr_notes = []

    if (
        sales_col and profit_col
        and sales_col in df.columns
        and profit_col in df.columns
    ):
        corr = df[[sales_col, profit_col]].corr().iloc[0, 1]
        corr_notes.append(
            f"Sales and profit are positively correlated (r = {corr:.2f})."
        )

    if (
        profit_col
        and "discount" in numeric_cols
        and profit_col in df.columns
    ):
        corr = df[[profit_col, "discount"]].corr().iloc[0, 1]
        if corr < 0:
            corr_notes.append(
                f"Discounting negatively impacts profit (r = {corr:.2f})."
            )

    # -------------------------------------------------
    # Recommendations (Quantified)
    # -------------------------------------------------
    recommendations = []

    if "discount" in df.columns:
        high_discount_rate = (df["discount"] > 0.3).mean()
        recommendations.append(
            f"Approximately {high_discount_rate*100:.1f}% of orders have discounts above 30%. "
            f"Reducing high discounts could improve margins."
        )

    if snapshot.get("top_segment"):
        recommendations.append(
            f"Scale growth initiatives in the {snapshot['top_segment']} segment."
        )

    if snapshot.get("top_region"):
        recommendations.append(
            f"Focus regional optimization efforts in the {snapshot['top_region']} region."
        )

    # -------------------------------------------------
    # Data Quality & Risk Notes
    # -------------------------------------------------
    dq_notes = [
        f"Missing values detected in {df.isna().any().sum()} columns.",
        "Identifier-like fields (e.g., postal_code) were excluded from numeric analysis.",
        "Sales and profit distributions show right-skew, indicating potential outliers."
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

    # Page 1 — Executive Snapshot
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))

    kpi_rows = []
    items = list(snapshot["kpis"].items())
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
        f"The dataset shows strong performance driven by the "
        f"<b>{snapshot.get('top_segment', 'leading')}</b> segment and "
        f"<b>{snapshot.get('top_region', 'key')}</b> region."
    )
    story.append(Paragraph(summary, styles["BodyText"]))

    if "risk" in snapshot:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"• {snapshot['risk']}", styles["BodyText"]))

    story.append(PageBreak())

    # Key Insights
    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(f"• {ins}", styles["BodyText"]))

    story.append(PageBreak())

    # Visuals
    for img in images.values():
        if img.exists():
            story.append(Image(str(img), width=16 * cm, height=5 * cm))

    story.append(PageBreak())

    # Correlation Insights
    story.append(Paragraph("Correlation Insights", styles["Heading2"]))
    for c in corr_notes:
        story.append(Paragraph(f"• {c}", styles["BodyText"]))

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
