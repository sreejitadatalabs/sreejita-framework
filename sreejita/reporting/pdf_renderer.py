from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.units import inch
from reportlab.lib import utils
from reportlab.graphics.shapes import Drawing, PolyLine, Circle


# =====================================================
# PAYLOAD NORMALIZER
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}

    if len(payload) == 1:
        payload = next(iter(payload.values()), payload)

    executive = payload.get("executive", {}) or {}

    return {
        "executive_snapshot": executive.get("snapshot", {}),
        "primary_kpis": executive.get("primary_kpis", []),
        "board_readiness": executive.get("board_readiness", {}),
        "board_readiness_trend": executive.get("board_readiness_trend", {}),
        "board_readiness_history": executive.get("board_readiness_history", []),
        "visuals": payload.get("visuals", []),
        "insights": payload.get("insights", []),
        "recommendations": payload.get("recommendations", []),
    }


# =====================================================
# FORMATTERS
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


def confidence_badge(conf: float | None) -> str:
    if conf is None:
        return "Unknown"
    if conf >= 0.85:
        return "🟢 High"
    if conf >= 0.70:
        return "🟡 Medium"
    return "🔴 Low"


def confidence_color(conf: float | None):
    if conf is None:
        return white
    if conf >= 0.85:
        return HexColor("#dcfce7")  # green
    if conf >= 0.70:
        return HexColor("#fef9c3")  # yellow
    return HexColor("#fee2e2")      # red


# =====================================================
# SPARKLINE (BOARD READINESS)
# =====================================================

def board_readiness_sparkline(values: List[int], width=120, height=30) -> Drawing:
    if not values or len(values) < 2:
        return Drawing(width, height)

    min_v, max_v = min(values), max(values)
    spread = max(max_v - min_v, 1)

    points = []
    step_x = width / (len(values) - 1)

    for i, v in enumerate(values):
        x = i * step_x
        y = ((v - min_v) / spread) * (height - 6) + 3
        points.append((x, y))

    d = Drawing(width, height)
    d.add(PolyLine(points, strokeColor=HexColor("#2563eb"), strokeWidth=2))
    d.add(Circle(points[-1][0], points[-1][1], 2, fillColor=HexColor("#2563eb")))
    return d


# =====================================================
# EXECUTIVE PDF RENDERER
# =====================================================

class ExecutivePDFRenderer:
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        payload = normalize_pdf_payload(payload)

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

        styles.add(ParagraphStyle("ExecTitle", fontSize=22, alignment=TA_CENTER))
        styles.add(ParagraphStyle("ExecSection", fontSize=15, spaceBefore=18, spaceAfter=10))
        styles.add(ParagraphStyle("ExecBody", fontSize=11, leading=15))
        styles.add(ParagraphStyle("ExecCaption", fontSize=9, textColor=HexColor("#6b7280")))

        # =================================================
        # COVER
        # =================================================
        br = payload["board_readiness"]

        story.append(Paragraph("SREEJITA INTELLIGENCE FRAMEWORK™", styles["ExecTitle"]))
        story.append(Paragraph("Executive Performance Report", styles["ExecSection"]))
        story.append(Paragraph(
            f"Board Readiness: <b>{br.get('score','-')}</b> / 100 "
            f"({br.get('band','-')})<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["ExecBody"]
        ))
        story.append(PageBreak())

        # =================================================
        # BOARD READINESS + SPARKLINE
        # =================================================
        story.append(Paragraph("Board Readiness Trend", styles["ExecSection"]))

        spark = board_readiness_sparkline(payload.get("board_readiness_history", []))
        story.append(spark)
        story.append(Spacer(1, 12))

        # =================================================
        # KPIs WITH CONFIDENCE HEAT
        # =================================================
        rows = [["Metric", "Value", "Confidence"]]
        styles_row = []

        for idx, k in enumerate(payload["primary_kpis"][:5], start=1):
            conf = k.get("confidence")
            rows.append([
                k["name"],
                format_kpi_value(k["name"], k["value"]),
                confidence_badge(conf),
            ])
            styles_row.append((
                "BACKGROUND",
                (0, idx),
                (-1, idx),
                confidence_color(conf)
            ))

        table = Table(rows, colWidths=[3.5 * inch, 2 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
            ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            *styles_row
        ]))

        story.append(Paragraph("Key Performance Indicators", styles["ExecSection"]))
        story.append(table)

        doc.build(story)
        return output_path
