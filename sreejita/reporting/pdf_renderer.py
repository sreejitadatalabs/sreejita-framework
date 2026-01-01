from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Flowable,
)
from reportlab.lib.units import inch


# =====================================================
# PAYLOAD NORMALIZER (DEFENSIVE, CONTRACT-SAFE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts:
    - domain-wrapped payload
    - hybrid.run() payload
    - raw executive payload

    Returns a flat, renderer-safe structure.
    """
    payload = payload if isinstance(payload, dict) else {}

    # If domain-wrapped ({"healthcare": {...}})
    if len(payload) == 1:
        payload = next(iter(payload.values()), payload)

    executive = payload.get("executive", payload)

    return {
        "snapshot": executive.get("snapshot", {}),
        "primary_kpis": executive.get("primary_kpis", []),
        "board_readiness": executive.get("board_readiness", {}),
        "board_readiness_trend": executive.get("board_readiness_trend", {}),
        "board_readiness_history": executive.get("board_readiness_history", []),
    }


# =====================================================
# FORMATTERS
# =====================================================

def format_kpi_value(name: str, value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, (int, float)):
        key = name.lower()

        if "rate" in key or "ratio" in key:
            return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"

        if "los" in key or "duration" in key or "days" in key:
            return f"{value:.1f} days"

        if "cost" in key or "amount" in key:
            if value >= 1_000_000:
                return f"${value / 1_000_000:.1f}M"
            if value >= 1_000:
                return f"${value / 1_000:.0f}K"
            return f"${value:,.0f}"

        return f"{value:,.0f}"

    return str(value)


def confidence_badge(conf: Optional[float]) -> str:
    if conf is None:
        return "—"
    if conf >= 0.85:
        return "🟢 High"
    if conf >= 0.70:
        return "🟡 Medium"
    return "🔴 Low"


def confidence_color(conf: Optional[float]):
    if conf is None:
        return white
    if conf >= 0.85:
        return HexColor("#dcfce7")
    if conf >= 0.70:
        return HexColor("#fef9c3")
    return HexColor("#fee2e2")


# =====================================================
# BOARD READINESS SPARKLINE (FLOWABLE SAFE)
# =====================================================

class BoardReadinessSparkline(Flowable):
    def __init__(self, values: List[int], width=120, height=30):
        super().__init__()
        self.values = values or []
        self.width = width
        self.height = height

    def draw(self):
        if len(self.values) < 2:
            return

        min_v = min(self.values)
        max_v = max(self.values)
        spread = max(max_v - min_v, 1)

        step_x = self.width / (len(self.values) - 1)
        points = []

        for i, v in enumerate(self.values):
            x = i * step_x
            y = ((v - min_v) / spread) * (self.height - 6) + 3
            points.append((x, y))

        self.canv.setStrokeColor(HexColor("#2563eb"))
        self.canv.setLineWidth(2)

        for i in range(len(points) - 1):
            self.canv.line(
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1]
            )

        # last point
        self.canv.circle(points[-1][0], points[-1][1], 2, stroke=0, fill=1)


# =====================================================
# EXECUTIVE PDF RENDERER (AUTHORITATIVE)
# =====================================================

class ExecutivePDFRenderer:
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        payload = normalize_pdf_payload(payload)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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

        # -------------------------------------------------
        # STYLES
        # -------------------------------------------------
        styles.add(ParagraphStyle(
            "ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            spaceAfter=18,
        ))
        styles.add(ParagraphStyle(
            "ExecSection",
            fontSize=15,
            fontName="Helvetica-Bold",
            spaceBefore=18,
            spaceAfter=10,
        ))
        styles.add(ParagraphStyle(
            "ExecBody",
            fontSize=11,
            leading=15,
        ))

        # =================================================
        # COVER PAGE
        # =================================================
        br = payload.get("board_readiness", {})

        story.append(Paragraph(
            "SREEJITA INTELLIGENCE FRAMEWORK™",
            styles["ExecTitle"]
        ))
        story.append(Paragraph(
            "Executive Performance Report",
            styles["ExecSection"]
        ))
        story.append(Paragraph(
            f"<b>Board Readiness:</b> {br.get('score','-')} / 100 "
            f"({br.get('band','-')})<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["ExecBody"]
        ))

        story.append(PageBreak())

        # =================================================
        # BOARD READINESS TREND
        # =================================================
        story.append(Paragraph(
            "Board Readiness Trend",
            styles["ExecSection"]
        ))

        history = payload.get("board_readiness_history", [])
        story.append(BoardReadinessSparkline(history))
        story.append(Spacer(1, 12))

        # =================================================
        # KPI TABLE WITH CONFIDENCE HEAT
        # =================================================
        primary = payload.get("primary_kpis", [])

        if primary:
            rows = [["Metric", "Value", "Confidence"]]
            row_styles = []

            for idx, kpi in enumerate(primary[:5], start=1):
                conf = kpi.get("confidence")
                rows.append([
                    kpi.get("name", "Metric"),
                    format_kpi_value(
                        kpi.get("name", ""),
                        kpi.get("value")
                    ),
                    confidence_badge(conf),
                ])
                row_styles.append((
                    "BACKGROUND",
                    (0, idx),
                    (-1, idx),
                    confidence_color(conf)
                ))

            table = Table(
                rows,
                colWidths=[3.5 * inch, 2 * inch, 1.5 * inch]
            )

            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                *row_styles,
            ]))

            story.append(Paragraph(
                "Key Performance Indicators",
                styles["ExecSection"]
            ))
            story.append(table)

        doc.build(story)
        return output_path
