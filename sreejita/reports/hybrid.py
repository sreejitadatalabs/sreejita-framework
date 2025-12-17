from pathlib import Path
from datetime import datetime, timezone
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

from sreejita.domains.router import decide_domain, apply_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.reporting.orchestrator import generate_report_payload
from sreejita.reporting.formatters import fmt_currency, fmt_percent

from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.categorical import bar
from sreejita.visuals.correlation import shipping_cost_vs_sales


# =========================================================
# SAFE DATA LOADER (CSV + EXCEL)
# =========================================================
def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")

    if suffix in [".xls", ".xlsx"]:
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise RuntimeError(
                "Excel support requires 'openpyxl'. "
                "Install with: pip install openpyxl"
            ) from e

    raise ValueError(f"Unsupported file type: {suffix}")


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
# MAIN ENTRY
# =========================================================
def run(
    input_path: str,
    config: dict,
    output_path: Optional[str] = None,
) -> str:

    input_path = Path(input_path)

    if output_path is None:
        out_dir = input_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / (
            f"Hybrid_Report_v3_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    # -------------------------------
    # LOAD DATA (SAFE)
    # -------------------------------
    df = load_dataframe(input_path)

    # -------------------------------
    # DOMAIN & POLICY
    # -------------------------------
    decision = decide_domain(df)
    policy = PolicyEngine().evaluate(decision)

    if "domain" in config:
        df = apply_domain(df, config["domain"]["name"])

    # -------------------------------
    # INTELLIGENCE PAYLOAD
    # -------------------------------
    payload = generate_report_payload(df, decision, policy)

    kpis = payload.get("kpis", {})
    insights = payload.get("insights", [])
    recommendations = payload.get("recommendations", [])
    dq = payload.get("data_quality", {})

    # -------------------------------
    # VISUALS
    # -------------------------------
    visuals = []
    img_dir = Path(output_path).parent

    sales_col = config.get("dataset", {}).get("sales", "sales")
    date_col = config.get("dataset", {}).get("date")
    shipping_col = config.get("dataset", {}).get("shipping", "shipping_cost")

    # 1️⃣ Sales trend
    if date_col:
        trend_path = img_dir / "sales_trend.png"
        plot_monthly(df, date_col, sales_col, trend_path)
        if trend_path.exists():
            visuals.append((str(trend_path), "Sales trend over time."))

    # 2️⃣ Category performance
    cat_path = img_dir / "sales_by_category.png"
    bar(df, "category", cat_path)
    if cat_path.exists():
        visuals.append((str(cat_path), "Revenue contribution by category."))

    # 3️⃣ Shipping vs Sales scatter (visual polish)
    scatter_path = img_dir / "shipping_cost_vs_sales.png"
    try:
        shipping_cost_vs_sales(
            df=df,
            sales_col=sales_col,
            shipping_col=shipping_col,
            out=scatter_path,
        )
        if scatter_path.exists() and scatter_path.stat().st_size > 0:
            visuals.append(
                (
                    str(scatter_path),
                    "Shipping cost efficiency varies by product category.",
                )
            )
    except Exception:
        pass  # polish visuals must never break report

    # =====================================================
    # PDF BUILD
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

    # =====================================================
    # PAGE 1 — EXECUTIVE BRIEF
    # =====================================================
    story.append(Paragraph("EXECUTIVE BRIEF (1-MINUTE READ)", title))
    story.append(Spacer(1, 12))

    quick_wins = sum(r.get("impact", 0) for r in recommendations if r.get("impact"))

    story.append(
        Paragraph(
            f"""
            💰 Revenue Status: {fmt_currency(kpis.get("total_sales"))}<br/>
            ⚠️ Issues Found: {len([i for i in insights if i.get("severity") == "WARNING"])} WARNINGs<br/>
            💡 Available Quick Wins: {fmt_currency(quick_wins)} annually<br/>
            ✅ Data Quality: {dq.get("confidence", "High")}<br/>
            ⏰ Next Step: Initiate shipping audit (5–7 days)
            """,
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 16))

    # KPI TABLE (SEMANTIC)
    formatted_kpis = []
    for k, v in kpis.items():
        key = k.replace("_", " ").title()
        kl = k.lower()

        if "ratio" in kl or "margin" in kl:
            val = fmt_percent(v)
        elif "count" in kl or "orders" in kl or "records" in kl:
            val = f"{int(v):,}"
        else:
            val = fmt_currency(v)

        formatted_kpis.append([key, val])

    table = Table(formatted_kpis, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "#999999")]))
    story.append(table)

    story.append(PageBreak())

    # =====================================================
    # PAGE 2 — EVIDENCE SNAPSHOT
    # =====================================================
    story.append(Paragraph("EVIDENCE SNAPSHOT", title))
    story.append(Spacer(1, 12))

    for img, cap in visuals:
        story.append(Image(img, width=14 * cm, height=8 * cm))
        story.append(Spacer(1, 6))
        story.append(Paragraph(cap, styles["BodyText"]))
        story.append(Spacer(1, 18))

    story.append(PageBreak())

    # =====================================================
    # PAGE 3 — INSIGHTS
    # =====================================================
    story.append(Paragraph("KEY INSIGHTS", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(f"• {ins.get('message')}", styles["BodyText"]))

    story.append(PageBreak())

    # =====================================================
    # PAGE 4 — RECOMMENDATIONS
    # =====================================================
    story.append(Paragraph("RECOMMENDATIONS", styles["Heading2"]))
    for r in recommendations:
        story.append(Paragraph(f"<b>Action:</b> {r.get('action')}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Impact:</b> {fmt_currency(r.get('impact'))}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Timeline:</b> {r.get('timeline')}", styles["BodyText"]))
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # =====================================================
    # PAGE 5 — DATA QUALITY & METADATA
    # =====================================================
    story.append(Paragraph("DATA QUALITY & CONFIDENCE", styles["Heading2"]))
    for k, v in dq.items():
        story.append(Paragraph(f"{k.title()}: {v}", styles["BodyText"]))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            f"""
            Report Version: v3.2 (Retail v2.8.2)<br/>
            Dataset: {input_path.name}<br/>
            Records Analyzed: {len(df):,}<br/>
            Generated At: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            """,
            styles["BodyText"],
        )
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    return str(output_path)
