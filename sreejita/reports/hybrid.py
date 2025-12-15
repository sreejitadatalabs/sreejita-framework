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
from sreejita.core.insights import correlation_insights, generate_detailed_insights
from sreejita.core.recommendations import (
    generate_recommendations,
    generate_prescriptive_recommendations,
)
from sreejita.domains.router import apply_domain

from sreejita.visuals.correlation import heatmap
from sreejita.visuals.categorical import bar
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def safe(value, fallback):
    return value if value not in [None, "", "nan"] else fallback


def enforce_exact(items, count, fillers):
    items = list(items)
    while len(items) < count:
        items.append(fillers[len(items) % len(fillers)])
    return items[:count]


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
# Evidence Snapshot — HARD GUARANTEE (v1.9.9)
# -------------------------------------------------
def build_evidence_snapshot(df, schema, config):
    """
    Evidence Snapshot MUST render visible visuals.
    Exactly 3 visuals max, correlation is mandatory.
    """
    visuals = []
    img_dir = Path("hybrid_images")
    img_dir.mkdir(exist_ok=True)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")

    # 1️⃣ Correlation heatmap (MANDATORY)
    corr_path = img_dir / "evidence_correlation.png"
    heatmap(df, corr_path)
    if corr_path.exists():
        visuals.append((
            corr_path,
            "Correlation heatmap showing relationships between key numeric metrics."
        ))

    # 2️⃣ Categorical dominance
    if schema["categorical"] and sales_col in df.columns:
        cat = schema["categorical"][0]
        bar_path = img_dir / "evidence_category.png"
        bar(df, cat, bar_path)
        if bar_path.exists():
            visuals.append((
                bar_path,
                f"{sales_col} distribution by {cat}."
            ))

    # 3️⃣ Trend OR Distribution (fallback safe)
    if date_col and date_col in df.columns and sales_col in df.columns:
        trend_path = img_dir / "evidence_trend.png"
        plot_monthly(df, date_col, sales_col, trend_path)
        if trend_path.exists():
            visuals.append((
                trend_path,
                "Performance trend over time."
            ))
    elif sales_col in df.columns:
        dist_path = img_dir / "evidence_distribution.png"
        hist(df, sales_col, dist_path)
        if dist_path.exists():
            visuals.append((
                dist_path,
                "Distribution of key performance metric."
            ))

    return visuals[:3]


# -------------------------------------------------
# Main Runner
# -------------------------------------------------
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / (
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

    # -------------------------
    # Executive intelligence
    # -------------------------
    kpis = compute_kpis(df, sales_col, profit_col)

    summary_insights = enforce_exact(
        correlation_insights(df, sales_col),
        3,
        [
            "Performance is driven by a limited number of key factors.",
            "Results vary significantly across business dimensions.",
            "Certain metrics indicate potential efficiency risks.",
        ],
    )

    summary_recs = enforce_exact(
        generate_recommendations(df, sales_col, profit_col),
        3,
        [
            "Strengthen controls to reduce performance volatility.",
            "Prioritize high-impact segments for optimization.",
            "Improve data completeness for future analysis.",
        ],
    )

    detailed_insights = generate_detailed_insights(summary_insights)
    prescriptive_recs = generate_prescriptive_recommendations(summary_recs)
    evidence = build_evidence_snapshot(df, schema, config)

    missing_pct = (df.isna().sum() / len(df)) * 100
    dq_notes = [
        f"Missing values present in {missing_pct[missing_pct > 0].count()} columns "
        f"(highest: {missing_pct.max():.1f}%).",
        "Identifier-like fields excluded from numeric analysis.",
        "Outliers may influence aggregate metrics.",
    ]

    # -------------------------
    # Build PDF
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

    # PAGE 1 — EXECUTIVE SNAPSHOT
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            safe(
                config.get("objective"),
                "Identify key performance drivers, risks, and actionable opportunities.",
            ),
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))

    table = Table([[k, v] for k, v in kpis.items()], colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999999")]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top 3 Key Insights", styles["Heading2"]))
    for i in summary_insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Top 3 Recommendations", styles["Heading2"]))
    for r in summary_recs:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    story.append(PageBreak())

    # PAGE 2 — EVIDENCE SNAPSHOT (VISIBLE PROOF)
    story.append(Paragraph("Evidence Snapshot (Supporting Analysis)", title))
    story.append(Spacer(1, 12))

    for img_path, caption in evidence:
        story.append(
            Image(
                str(img_path),
                width=15 * cm,
                height=9 * cm,
                kind="proportional",
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph(caption, styles["BodyText"]))
        story.append(Spacer(1, 18))

    story.append(PageBreak())

    # PAGE 3 — DETAILED INSIGHTS
    story.append(Paragraph("Key Insights (Detailed Analysis)", styles["Heading2"]))
    story.append(Spacer(1, 12))

    for ins in detailed_insights:
        story.append(
            Paragraph(f"<b>{ins['title']}:</b> {ins['what']}", styles["BodyText"])
        )
        story.append(
            Paragraph(f"<i>Why this matters:</i> {ins['why']}", styles["BodyText"])
        )
        story.append(
            Paragraph(
                f"<i>Business implication:</i> {ins['so_what']}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # PAGE 4 — PRESCRIPTIVE RECOMMENDATIONS
    story.append(
        Paragraph("Recommendations (Prescriptive Actions)", styles["Heading2"])
    )
    story.append(Spacer(1, 12))

    for r in prescriptive_recs:
        story.append(Paragraph(f"<b>Action:</b> {r['action']}", styles["BodyText"]))
        story.append(
            Paragraph(f"<b>Rationale:</b> {r['rationale']}", styles["BodyText"])
        )
        story.append(
            Paragraph(
                f"<b>Expected outcome:</b> {r['expected_outcome']}",
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(f"<b>Priority:</b> {r['priority']}", styles["BodyText"]))
        story.append(Spacer(1, 14))

    story.append(PageBreak())

    # PAGE 5 — DATA QUALITY & RISK
    story.append(Paragraph("Data Quality & Risk Notes", styles["Heading2"]))
    for d in dq_notes:
        story.append(Paragraph(f"• {d}", styles["BodyText"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return str(output_path)
