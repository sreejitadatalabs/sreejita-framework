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
    Image,
    Flowable,
)
from reportlab.lib.units import inch


# =====================================================
# PAYLOAD NORMALIZER (DEFENSIVE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload or {}

    # Hybrid payload
    if "executive" in payload:
        exec_ = payload["executive"]
        return {
            "snapshot": exec_.get("snapshot", {}),
            "primary_kpis": exec_.get("primary_kpis", []),
            "board_readiness": exec_.get("board_readiness", {}),
            "board_readiness_trend": exec_.get("board_readiness_trend", {}),
            "board_readiness_history": exec_.get("board_readiness_history", []),
            "insights": payload.get("insights", []),
            "recommendations": payload.get("recommendations", []),
            "visuals": payload.get("visuals", []),
        }

    # Domain wrapped
    if len(payload) == 1:
        payload = next(iter(payload.values()))

    exec_ = payload.get("executive", {})
    return {
        "snapshot": exec_.get("snapshot", {}),
        "primary_kpis": exec_.get("primary_kpis", []),
        "board_readiness": exec_.get("board_readiness", {}),
        "board_readiness_trend": exec_.get("board_readiness_trend", {}),
        "board_readiness_history": exec_.get("board_readiness_history", []),
        "insights": payload.get("insights", []),
        "recommendations": payload.get("recommendations", []),
        "visuals": payload.get("visuals", []),
    }


# =====================================================
# CONFIDENCE HELPERS
# =====================================================

def confidence_badge(conf: Optional[float]) -> str:
    if conf is None:
        return "—"
    if conf >= 0.85:
        return "High"
    if conf >= 0.70:
        return "Medium"
    return "Low"


def confidence_color(conf: Optional[float]):
    if conf is None:
        return white
    if conf >= 0.85:
        return HexColor("#dcfce7")
    if conf >= 0.70:
        return HexColor("#fef9c3")
    return HexColor("#fee2e2")


# =====================================================
# BOARD READINESS SPARKLINE
# =====================================================

class BoardReadinessSparkline(Flowable):
    def __init__(self, values: List[int], width=140, height=32):
        super().__init__()
        self.values = values or []
        self.width = width
        self.height = height

    def draw(self):
        if len(self.values) < 2:
            return

        min_v, max_v = min(self.values), max(self.values)
        spread = max(max_v - min_v, 1)
        step = self.width / (len(self.values) - 1)

        points = [
            (i * step, ((v - min_v) / spread) * (self.height - 6) + 3)
            for i, v in enumerate(self.values)
        ]

        self.canv.setStrokeColor(HexColor("#2563eb"))
        self.canv.setLineWidth(2)

        for i in range(len(points) - 1):
            self.canv.line(*points[i], *points[i + 1])

        self.canv.circle(points[-1][0], points[-1][1], 2, stroke=0, fill=1)


# =====================================================
# EXECUTIVE PDF RENDERER
# =====================================================

class ExecutivePDFRenderer:
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        data = normalize_pdf_payload(payload)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle("Title", fontSize=22, alignment=TA_CENTER, fontName="Helvetica-Bold"))
        styles.add(ParagraphStyle("Section", fontSize=15, spaceBefore=18, fontName="Helvetica-Bold"))
        styles.add(ParagraphStyle("Body", fontSize=11, leading=14))

        story: List[Any] = []

        # =================================================
        # COVER
        # =================================================
        br = data["board_readiness"]
        story.append(Paragraph("SREEJITA INTELLIGENCE FRAMEWORK™", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"<b>Board Readiness:</b> {br.get('score','-')} / 100 "
            f"({br.get('band','-')})<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["Body"]
        ))
        story.append(PageBreak())

        # =================================================
        # BOARD READINESS TREND
        # =================================================
        story.append(Paragraph("Board Readiness Trend", styles["Section"]))
        story.append(BoardReadinessSparkline(data["board_readiness_history"]))
        story.append(PageBreak())

        # =================================================
        # KPI TABLE
        # =================================================
        if data["primary_kpis"]:
            rows = [["Metric", "Value", "Confidence"]]
            row_styles = []

            for idx, k in enumerate(data["primary_kpis"][:5], start=1):
                conf = k.get("confidence")
                rows.append([k["name"], str(k["value"]), confidence_badge(conf)])
                row_styles.append(("BACKGROUND", (0, idx), (-1, idx), confidence_color(conf)))

            table = Table(rows, colWidths=[3.5 * inch, 2 * inch, 1.5 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                *row_styles,
            ]))

            story.append(Paragraph("Key Performance Indicators", styles["Section"]))
            story.append(table)
            story.append(PageBreak())

        # =================================================
        # INSIGHTS
        # =================================================
        insights = data["insights"]
        if insights:
            story.append(Paragraph("Key Insights", styles["Section"]))
            for i in insights[:8]:
                story.append(Paragraph(
                    f"<b>{i.get('level','INFO')}</b>: {i.get('title','')}<br/>{i.get('so_what','')}",
                    styles["Body"]
                ))
                story.append(Spacer(1, 6))
            story.append(PageBreak())

        # =================================================
        # RECOMMENDATIONS
        # =================================================
        recs = data["recommendations"]
        if recs:
            story.append(Paragraph("Executive Recommendations", styles["Section"]))
            for r in recs[:7]:
                story.append(Paragraph(
                    f"<b>{r.get('priority','')}</b> — {r.get('action','')}<br/>"
                    f"Owner: {r.get('owner','')} | Timeline: {r.get('timeline','')}",
                    styles["Body"]
                ))
                story.append(Spacer(1, 6))
            story.append(PageBreak())

        # =================================================
        # VISUAL EVIDENCE
        # =================================================
        visuals = data["visuals"]
        if visuals:
            story.append(Paragraph("Visual Evidence", styles["Section"]))
            for v in visuals[:6]:
                try:
                    img = Image(v["path"], width=5.5 * inch, height=3 * inch)
                    story.append(img)
                    story.append(Paragraph(v.get("caption",""), styles["Body"]))
                    story.append(Spacer(1, 12))
                except Exception:
                    continue

        doc.build(story)
        return output_path
