import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple

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

# -------------------------
# v2.4 / v2.5 / v2.6
# -------------------------
from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.domains.router import decide_domain, apply_domain
from sreejita.policy.engine import PolicyEngine

# -------------------------
# v1 core (UNCHANGED)
# -------------------------
from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.schema import detect_schema
from sreejita.core.insights import correlation_insights, generate_detailed_insights
from sreejita.core.recommendations import (
    generate_recommendations,
    generate_prescriptive_recommendations,
)

# -------------------------
# visuals (UNCHANGED)
# -------------------------
from sreejita.visuals.correlation import heatmap
from sreejita.visuals.categorical import bar
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist


# =========================================================
# HARD SAFETY — VISUAL CERTAINTY
# =========================================================
def assert_image_valid(path: Path):
    if not path.exists():
        raise RuntimeError(f"Evidence image missing: {path}")
    if path.stat().st_size < 5_000:
        raise RuntimeError(f"Evidence image invalid or empty: {path}")


def enforce_exact(items, n, fillers):
    items = list(items)
    while len(items) < n:
        items.append(fillers[len(items) % len(fillers)])
    return items[:n]


def safe(val, fallback):
    return val if val not in [None, "", "nan"] else fallback


# =========================================================
# HEADER / FOOTER
# =========================================================
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


# =========================================================
# DATA LOADER
# =========================================================
def load_dataframe(path: Path):
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")

    elif path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path)

    else:
        raise ValueError("Unsupported file type")


# =========================================================
# EVIDENCE SNAPSHOT (UNCHANGED)
# =========================================================
def build_evidence_snapshot(
    df: pd.DataFrame, schema: dict, config: dict
) -> List[Tuple[Path, str]]:

    visuals = []
    img_dir = Path("hybrid_images").resolve()
    img_dir.mkdir(exist_ok=True)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")

    corr_path = img_dir / "evidence_correlation.png"
    heatmap(df, corr_path)
    assert_image_valid(corr_path)
    visuals.append((corr_path, "Correlation heatmap showing relationships between key numeric metrics."))

    if schema["categorical"] and sales_col in df.columns:
        cat = schema["categorical"][0]
        bar_path = img_dir / "evidence_category.png"
        bar(df, cat, bar_path)
        assert_image_valid(bar_path)
        visuals.append((bar_path, f"{sales_col} distribution by {cat}."))

    if date_col and date_col in df.columns and sales_col in df.columns:
        trend_path = img_dir / "evidence_trend.png"
        plot_monthly(df, date_col, sales_col, trend_path)
        assert_image_valid(trend_path)
        visuals.append((trend_path, "Performance trend over time."))
    elif sales_col in df.columns:
        dist_path = img_dir / "evidence_distribution.png"
        hist(df, sales_col, dist_path)
        assert_image_valid(dist_path)
        visuals.append((dist_path, "Distribution of key performance metric."))

    return visuals[:3]


# =========================================================
# MAIN REPORT PIPELINE (v2.6 ENABLED)
# =========================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_v3_1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    df_raw = load_dataframe(input_path)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    df = clean_dataframe(df_raw, [date_col] if date_col else None)["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    schema = detect_schema(df)

    # =====================================================
    # 🔥 v2.4 + v2.5 + v2.6 INTELLIGENCE
    # =====================================================
    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    v26_payload = generate_report_payload(
        df=df,
        decision=decision,
        policy=policy
    )

    # =====================================================
    # CONTENT SELECTION (v2.6 → v1 fallback)
    # =====================================================
    if v26_payload:
        kpis = v26_payload["kpis"]
        top_insights = [i["title"] for i in v26_payload["insights"]][:3]
        top_recs = [r["action"] for r in v26_payload["recommendations"]][:3]
        detailed_insights = generate_detailed_insights(top_insights)
        prescriptive_recs = generate_prescriptive_recommendations(top_recs)
    else:
        kpis = compute_kpis(df, sales_col, profit_col)
        top_insights = enforce_exact(
            correlation_insights(df, sales_col),
            3,
            [
                "Performance is driven by a limited number of dominant factors.",
                "Results vary significantly across business dimensions.",
                "Certain metrics indicate efficiency risks.",
            ],
        )
        top_recs = enforce_exact(
            generate_recommendations(df, sales_col, profit_col),
            3,
            [
                "Reduce volatility through tighter operational controls.",
                "Prioritize high-impact segments for optimization.",
                "Improve data completeness for future decision-making.",
            ],
        )
        detailed_insights = generate_detailed_insights(top_insights)
        prescriptive_recs = generate_prescriptive_recommendations(top_recs)

    evidence = build_evidence_snapshot(df, schema, config)

    if v26_payload and v26_payload.get("visuals"):
        for v in v26_payload["visuals"]:
            evidence.append((v["path"], v["caption"]))

    missing_pct = (df.isna().sum() / len(df)) * 100
    dq_notes = [
        f"Missing values detected in {missing_pct[missing_pct > 0].count()} columns (highest: {missing_pct.max():.1f}%).",
        "Identifier-like fields excluded from numeric analysis.",
        "Outliers may influence aggregate metrics.",
    ]

    # =====================================================
    # PDF BUILD (UNCHANGED)
    # =====================================================
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
    for i in top_insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Top 3 Recommendations", styles["Heading2"]))
    for r in top_recs:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    story.append(PageBreak())

    story.append(Paragraph("Evidence Snapshot (Supporting Analysis)", title))
    for img_path, caption in evidence:
        story.append(Image(str(img_path), width=14 * cm, height=8 * cm))
        story.append(Spacer(1, 6))
        story.append(Paragraph(caption, styles["BodyText"]))
        story.append(Spacer(1, 18))

    story.append(PageBreak())

    story.append(Paragraph("Key Insights (Detailed Analysis)", styles["Heading2"]))
    for ins in detailed_insights:
        story.append(Paragraph(f"<b>{ins['title']}:</b> {ins['what']}", styles["BodyText"]))
        story.append(Paragraph(f"<i>Why this matters:</i> {ins['why']}", styles["BodyText"]))
        story.append(Paragraph(f"<i>Business implication:</i> {ins['so_what']}", styles["BodyText"]))
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    story.append(Paragraph("Recommendations (Prescriptive Actions)", styles["Heading2"]))
    for r in prescriptive_recs:
        story.append(Paragraph(f"<b>Action:</b> {r['action']}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Rationale:</b> {r['rationale']}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Expected outcome:</b> {r['expected_outcome']}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Priority:</b> {r['priority']}", styles["BodyText"]))
        story.append(Spacer(1, 14))

    story.append(PageBreak())

    story.append(Paragraph("Data Quality & Risk Notes", styles["Heading2"]))
    for d in dq_notes:
        story.append(Paragraph(f"• {d}", styles["BodyText"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return str(output_path)
