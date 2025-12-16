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
# visuals (v1 fallback)
# -------------------------
from sreejita.visuals.correlation import heatmap
from sreejita.visuals.categorical import bar
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.distributions import hist


# =========================================================
# SAFETY
# =========================================================
def assert_image_valid(path: Path):
    if not path.exists():
        raise RuntimeError(f"Evidence image missing: {path}")
    if path.stat().st_size < 5_000:
        raise RuntimeError(f"Evidence image invalid or empty: {path}")


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
# GENERIC EVIDENCE (v1 fallback)
# =========================================================
def build_generic_evidence(df, schema, config):
    visuals = []
    img_dir = Path("hybrid_images").resolve()
    img_dir.mkdir(exist_ok=True)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")

    corr_path = img_dir / "evidence_correlation.png"
    heatmap(df, corr_path)
    assert_image_valid(corr_path)
    visuals.append((corr_path, "Correlation heatmap across numeric metrics."))

    if schema["categorical"] and sales_col in df.columns:
        bar_path = img_dir / "evidence_category.png"
        bar(df, schema["categorical"][0], bar_path)
        assert_image_valid(bar_path)
        visuals.append((bar_path, f"{sales_col} by category."))

    if date_col and sales_col in df.columns:
        trend_path = img_dir / "evidence_trend.png"
        plot_monthly(df, date_col, sales_col, trend_path)
        assert_image_valid(trend_path)
        visuals.append((trend_path, "Trend over time."))

    return visuals[:3]


# =========================================================
# v2.7 PRESENTATION HELPERS
# =========================================================
def render_executive_context(story, payload, styles):
    if not payload:
        return

    rows = [
        ["Detected Domain", payload["domain"].title()],
        ["Domain Confidence", f"{payload['domain_confidence']:.2f}"],
        ["Policy Status", payload["policy_status"]],
    ]

    table = Table(rows, colWidths=[6 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, "#999999"),
        ("BACKGROUND", (0, 0), (-1, 0), "#EEEEEE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))


def top_kpis(kpis, limit=4):
    return list(kpis.items())[:limit]


def render_insight_cards(story, insights, styles):
    for ins in insights:
        sev = ins.get("severity", "medium").upper()
        story.append(
            Paragraph(
                f"<b>[{sev}] {ins['title']}</b><br/><i>{ins.get('evidence','')}</i>",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 10))


def render_recommendation_table(story, recs):
    rows = [["Action", "Priority", "Expected Impact"]]
    for r in recs:
        rows.append([r["action"], r["priority"], r["expected_impact"]])

    table = Table(rows, colWidths=[8 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, "#999999"),
        ("BACKGROUND", (0, 0), (-1, 0), "#EEEEEE"),
    ]))
    story.append(table)


# =========================================================
# MAIN PIPELINE — v2.7
# =========================================================
def run(input_path: str, config: dict, output_path: Optional[str] = None) -> str:
    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"Hybrid_Report_v3_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    df_raw = load_dataframe(input_path)

    date_col = config.get("dataset", {}).get("date")
    sales_col = config.get("dataset", {}).get("sales")
    profit_col = config.get("dataset", {}).get("profit")

    df = clean_dataframe(df_raw, [date_col] if date_col else None)["df"]

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    schema = detect_schema(df)

    decision = decide_domain(df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    v26_payload = generate_report_payload(df, decision, policy)

    if v26_payload:
        kpis = v26_payload["kpis"]
        insights = v26_payload["insights"]
        recs = v26_payload["recommendations"]
        visuals = [(v["path"], v["caption"]) for v in v26_payload.get("visuals", [])]
    else:
        kpis = compute_kpis(df, sales_col, profit_col)
        insights = [{"title": t, "severity": "medium"} for t in correlation_insights(df, sales_col)]
        recs = generate_recommendations(df, sales_col, profit_col)
        visuals = build_generic_evidence(df, schema, config)

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

    # ---------------- PAGE 1 ----------------
    story.append(Paragraph("Executive Snapshot", title))
    story.append(Spacer(1, 12))
    render_executive_context(story, v26_payload, styles)

    story.append(Paragraph(
        safe(config.get("objective"),
             "Identify key performance drivers, risks, and opportunities."),
        styles["BodyText"])
    )
    story.append(Spacer(1, 12))

    kpi_table = Table(top_kpis(kpis), colWidths=[7 * cm, 7 * cm])
    kpi_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999999")]))
    story.append(kpi_table)
    story.append(PageBreak())

    # ---------------- PAGE 2 ----------------
    story.append(Paragraph("Evidence Snapshot", title))

    if visuals:
        for img, cap in visuals:
            story.append(Image(str(img), width=14 * cm, height=8 * cm))
            story.append(Spacer(1, 6))
            story.append(Paragraph(cap, styles["BodyText"]))
            story.append(Spacer(1, 18))
    else:
        story.append(
            Paragraph(
                "No anomalies detected. Baseline performance visuals shown.",
                styles["BodyText"]
            )
        )


    # ---------------- PAGE 3 ----------------
    story.append(Paragraph("Key Insights", title))
    render_insight_cards(story, insights[:5], styles)
    story.append(PageBreak())

    # ---------------- PAGE 4 ----------------
    story.append(Paragraph("Recommendations", title))
    render_recommendation_table(story, recs)
    story.append(PageBreak())

    # ---------------- PAGE 5 ----------------
    story.append(Paragraph("Data Quality & Risk Notes", styles["Heading2"]))
    missing_pct = (df.isna().sum() / len(df)) * 100
    story.append(Paragraph(
        f"Missing values present in {missing_pct[missing_pct > 0].count()} columns "
        f"(max {missing_pct.max():.1f}%).",
        styles["BodyText"]
    ))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return str(output_path)
