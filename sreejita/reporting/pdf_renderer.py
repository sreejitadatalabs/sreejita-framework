from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.units import inch
from reportlab.lib import utils


# =====================================================
# PAYLOAD NORMALIZER (STRICT & NESTED-SAFE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}

    if len(payload) == 1:
        payload = next(iter(payload.values()), payload)

    executive = payload.get("executive", {}) or {}

    return {
        "executive_snapshot": (
            payload.get("executive_snapshot")
            or executive.get("snapshot")
            or {}
        ),
        "primary_kpis": (
            payload.get("primary_kpis")
            or executive.get("primary_kpis", [])
        ),
        "board_readiness": (
            payload.get("board_readiness")
            or executive.get("board_readiness", {})
        ),
        "board_readiness_trend": executive.get("board_readiness_trend", {}),
        "visuals": payload.get("visuals", []),
        "insights": payload.get("insights", []),
        "recommendations": payload.get("recommendations", []),
    }


# =====================================================
# KPI FORMATTER
# =====================================================

def format_kpi_value(key: str, value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, (int, float)):
        k = key.lower()

        if "rate" in k:
            return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"
        if "los" in k or "duration" in k:
            return f"{value:.1f} days"
        if "cost" in k:
            if value >= 1_000_000:
                return f"${value / 1_000_000:.1f}M"
            if value >= 1_000:
                return f"${value / 1_000:.0f}K"
            return f"${value:,.0f}"

        return f"{value:,.0f}"

    return str(value)


# =====================================================
# CONFIDENCE BADGE (PDF SAFE)
# =====================================================

def confidence_badge(conf: float | None) -> str:
    if conf is None:
        return "Unknown"
    if conf >= 0.85:
        return "🟢 High Confidence"
    if conf >= 0.70:
        return "🟡 Moderate Confidence"
    return "🔴 Low Confidence"


def trend_label(symbol: str) -> str:
    return {
        "↑": "Improving",
        "↓": "Deteriorating",
        "→": "Stable",
    }.get(symbol, "Stable")


# =====================================================
# EXECUTIVE PDF RENDERER
# =====================================================

class ExecutivePDFRenderer:
    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        payload = normalize_pdf_payload(payload)

        payload["visuals"] = sorted(
            payload.get("visuals", []),
            key=lambda x: x.get("importance", 0),
            reverse=True,
        )

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        story: List[Any] = []

        def add_style(name, **kwargs):
            if name not in styles:
                styles.add(ParagraphStyle(name=name, **kwargs))

        add_style("ExecTitle", fontSize=22, alignment=TA_CENTER, spaceAfter=18, fontName="Helvetica-Bold")
        add_style("ExecSection", fontSize=15, spaceBefore=18, spaceAfter=10, fontName="Helvetica-Bold")
        add_style("ExecBody", fontSize=11, leading=15, spaceAfter=6)
        add_style("ExecCaption", fontSize=9, alignment=TA_CENTER, textColor=HexColor("#6b7280"))

        # =================================================
        # COVER PAGE
        # =================================================
        snapshot = payload["executive_snapshot"]
        board = payload.get("board_readiness", {})

        story.append(Paragraph("SREEJITA INTELLIGENCE FRAMEWORK™", styles["ExecTitle"]))
        story.append(Paragraph("Executive Performance Report", styles["ExecSection"]))

        story.append(Paragraph(
            f"Overall Risk: {snapshot.get('overall_risk','-')}<br/>"
            f"Board Readiness: {board.get('score','-')} / 100 ({board.get('band','-')})<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["ExecBody"],
        ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Prepared by: <b>Sreejita Data Labs</b>", styles["ExecBody"]))
        story.append(PageBreak())

        # =================================================
        # BOARD READINESS
        # =================================================
        trend = payload.get("board_readiness_trend", {})

        if board:
            story.append(Paragraph("Board Readiness Assessment", styles["ExecSection"]))
            story.append(Paragraph(
                f"Score: {board.get('score')} / 100<br/>"
                f"Status: {board.get('band')}<br/>"
                f"Trend: {trend.get('trend','→')} ({trend_label(trend.get('trend'))})<br/>"
                f"Previous Score: {trend.get('previous_score','N/A')}",
                styles["ExecBody"]
            ))
            story.append(PageBreak())

        # =================================================
        # KPIs
        # =================================================
        if payload["primary_kpis"]:
            rows = [["Metric", "Value", "Confidence"]]
            for k in payload["primary_kpis"][:5]:
                rows.append([
                    k.get("name"),
                    format_kpi_value(k.get("name",""), k.get("value")),
                    confidence_badge(k.get("confidence")),
                ])

            table = Table(rows, colWidths=[3.5 * inch, 2 * inch, 1.5 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
                ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("PADDING", (0,0), (-1,-1), 8),
            ]))

            story.append(Paragraph("Key Performance Indicators", styles["ExecSection"]))
            story.append(table)
            story.append(PageBreak())

        # =================================================
        # VISUALS / INSIGHTS / RECOMMENDATIONS
        # =================================================
        for section, items, renderer in [
            ("Key Insights & Risks", payload["insights"], lambda i:
                [Paragraph(f"<b>{i['level']}:</b> {i['title']}", styles["ExecBody"]),
                 Paragraph(i["so_what"], styles["ExecBody"])]
            ),
            ("Recommendations", payload["recommendations"], lambda r:
                [Paragraph(r["action"], styles["ExecBody"])]
            ),
        ]:
            if items:
                story.append(Paragraph(section, styles["ExecSection"]))
                for item in items[:5]:
                    for p in renderer(item):
                        story.append(p)
                story.append(PageBreak())

        doc.build(story)
        return output_path
