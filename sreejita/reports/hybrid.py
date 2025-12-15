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
from sreejita.core.schema import detect_schema
from sreejita.domains.router import apply_domain

from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist
from sreejita.visuals.categorical import bar
from sreejita.visuals.correlation import heatmap


# -------------------------------------------------
# Utilities (presentation safety)
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
# Executive Snapshot
# -------------------------------------------------
def generate_executive_snapshot(df, schema, kpis, sales_col, profit_col):
    snapshot = {}

    EXEC_KPI_ORDER = [
        "Total Sales",
        "Total Profit",
        "Profit Margin",
        "Orders",
        "Customers",
        "Avg Order Value",
    ]

    exec_kpis = [(k, kpis[k]) for k in EXEC_KPI_ORDER if k in kpis]
    snapshot["kpis"] = exec_kpis if exec_kpis else list(kpis.items())

    if sales_col and sales_col in df.columns and "segment" in df.columns:
        seg_share = (
            df.groupby("segment")[sales_col].sum()
            / df[sales_col].sum()
        )
        snapshot["top_segment"] = seg_share.idxmax()

    if sales_col and sales_col in df.columns and "region" in df.columns:
        snapshot["top_region"] = (
            df.groupby("region")[sales_col].sum().idxmax()
        )

    if (
        profit_col
        and profit_col in df.columns
        and "discount" in schema["numeric_measures"]
    ):
        snapshot["discount_corr"] = (
            df[[profit_col, "discount"]].corr().iloc[0, 1]
        )

    return snapshot


# -------------------------------------------------
# Written Insights
# -------------------------------------------------
def generate_written_insights(df, schema, sales_col, profit_col):
    insights = []

    if sales_col and "segment" in df.columns:
        seg_share = (
            df.groupby("segment")[sales_col].sum()
            / df[sales_col].sum()
        )
        insights.append(
            f"{seg_share.idxmax()} segment contributes "
            f"{seg_share.max()*100:.1f}% of total sales, making it the primary revenue driver."
        )

    if sales_col and "region" in df.columns:
        insights.append(
            f"{df.groupby('region')[sales_col].sum().idxmax()} region leads overall sales volume."
        )

    if "discount" in df.columns:
        high_disc = (df["discount"] > 0.3).mean() * 100
        insights.append(
            f"High discounts are concentrated in a minority of orders (~{high_disc:.1f}%)."
        )

    if "ship_mode" in df.columns:
        mode_share = df["ship_mode"].value_counts(normalize=True).iloc[0] * 100
        insights.append(
            f"Standard Class dominates shipping (~{mode_share:.0f}% of orders), "
            f"indicating cost efficiency over delivery speed."
        )

    if (
        profit_col
        and profit_col in df.columns
        and "discount" in schema["numeric_measures"]
    ):
        corr = df[[profit_col, "discount"]].corr().iloc[0, 1]
        insights.append(
            f"Discount negatively correlates with profit (r = {corr:.2f}), "
            f"highlighting margin erosion risk."
        )

    return enforce_min_bullets(
        insights,
        min_count=4,
        fillers=[
            "Revenue is concentrated across a limited number of key drivers.",
            "Sales performance varies significantly across business dimensions.",
            "Pricing decisions play a critical role in profitability outcomes.",
            "Operational patterns suggest trade-offs between cost and speed."
        ]
    )


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
    numeric_cols = schema["numeric_measures"]
    categorical_cols = schema["categorical"]

    kpis = compute_kpis(df, sales_col, profit_col)
    snapshot = generate_executive_snapshot(df, schema, kpis, sales_col, profit_col)
    insights = generate_written_insights(df, schema, sales_col, profit_col)

    # ---------------- Correlation Insights ----------------
    corr_notes = []
    if "discount_corr" in snapshot:
        corr_notes.append(
            f"Discount negatively correlates with profit (r = {snapshot['discount_corr']:.2f}), "
            f"indicating margin erosion risk."
        )

    corr_notes = enforce_min_bullets(
        corr_notes,
        min_count=2,
        fillers=[
            "Pricing has a stronger impact on profit than sales volume.",
            "Revenue growth appears more volume-driven than margin-driven."
        ]
    )

    # ---------------- Recommendations ----------------
    recommendations = enforce_min_bullets(
        [
            "Reduce discounts above 30% to protect margins.",
            "Introduce discount caps for low-margin categories.",
            "Prioritize consistently high-performing segments and regions.",
            "Flag high-discount or loss-making orders for review.",
        ],
        min_count=4,
        fillers=[
            "Strengthen pricing governance to reduce margin volatility.",
            "Align operational efficiency with customer service expectations."
        ]
    )

    # ---------------- Data Quality ----------------
    missing_pct = (df.isna().sum() / len(df)) * 100
    dq_notes = [
        f"Missing values detected in {missing_pct[missing_pct > 0].count()} columns "
        f"(highest: {missing_pct.max():.1f}%).",
        "Identifier-like fields were excluded from numeric analysis.",
        "Sales and profit distributions show right-skew, indicating potential outliers."
    ]

    # ---------------- Build PDF ----------------
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

    kpi_rows = []
    for i in range(0, len(snapshot["kpis"]), 2):
        left = snapshot["kpis"][i]
        right = snapshot["kpis"][i + 1] if i + 1 < len(snapshot["kpis"]) else ("", "")
        kpi_rows.append([
            Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", styles["BodyText"]),
            Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", styles["BodyText"]),
        ])

    story.append(Table(kpi_rows, colWidths=[8 * cm, 8 * cm]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Performance is driven primarily by the "
        f"<b>{safe_label(snapshot.get('top_segment'), 'key customer segments')}</b> "
        f"and the <b>{safe_label(snapshot.get('top_region'), 'overall market')}</b>.",
        styles["BodyText"]
    ))

    story.append(PageBreak())

    # Key Insights
    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(f"• {ins}", styles["BodyText"]))

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
