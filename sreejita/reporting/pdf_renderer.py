# sreejita/reporting/pdf_renderer.py

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
    Image,
    Table,
    TableStyle,
    PageBreak,
    Flowable,
)
from reportlab.lib.units import inch
from reportlab.lib import utils


# =====================================================
# PAYLOAD NORMALIZER (CONTRACT SAFE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    executive = payload.get("executive", {}) or {}

    return {
        "snapshot": executive.get("snapshot", {}),
        "primary_kpis": executive.get("primary_kpis", []),
        "board_readiness": executive.get("board_readiness", {}),
        "board_readiness_history": executive.get("board_readiness_history", []),
        "visuals": payload.get("visuals", []),
        "insights": payload.get("insights", []),
        "recommendations": payload.get("recommendations", []),
    }


# =====================================================
# FORMATTERS
# =====================================================

def format_value(key: str, value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, (int, float)):
        k = key.lower()
        if "rate" in k:
            return f"{value * 100:.1f}%"
        if "cost" in k:
            if value >= 1_000_000:
                return f"${value/1_000_000:.1f}M"
            if value >= 1_000:
                return f"${value/1_000:.0f}K"
            return f"${value:,.0f}"
        if "duration" in k or "los" in k:
            return f"{value:.1f} days"
        return f"{value:,.0f}"

    return str(value)


# =====================================================
# BOARD READINESS SPARKLINE
# =====================================================

class BoardReadinessSparkline(Flowable):
    def __init__(self, values: List[int], width=140, height=35):
        super().__init__()
        self.values = values
        self.width = width
        self.height = height

    def draw(self):
        if not self.values or len(self.values) < 2:
            return

        min_v, max_v = min(self.values), max(self.values)
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
            self.canv.line(*points[i], *points[i + 1])

        self.canv.circle(points[-1][0], points[-1][1], 2, stroke=0, fill=1)


# =====================================================
# EXECUTIVE PDF RENDERER (FINAL, SAFE)
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
        # SAFE CUSTOM STYLES (NO COLLISIONS)
        # -------------------------------------------------
        styles.add(ParagraphStyle(
            name="ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName="Helvetica-Bold",
        ))
        styles.add(ParagraphStyle(
            name="ExecSection",
            fontSize=15,
            spaceBefore=18,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        ))
        styles.add(ParagraphStyle(
            name="ExecBody",
            fontSize=11,
            leading=15,
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            name="ExecCaption",
            fontSize=9,
            textColor=HexColor("#6b7280"),
            alignment=TA_CENTER,
        ))

        # =================================================
        # COVER PAGE
        # =================================================
        br = payload["board_readiness"]

        story.append(Paragraph("Sreejita Executive Performance Report", styles["ExecTitle"]))
        story.append(Paragraph(
            f"<b>Board Readiness:</b> {br.get('score','-')} / 100 "
            f"({br.get('band','-')})<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["ExecBody"],
        ))

        story.append(PageBreak())

        # =================================================
        # BOARD READINESS TREND
        # =================================================
        story.append(Paragraph("Board Readiness Trend", styles["ExecSection"]))
        story.append(BoardReadinessSparkline(payload["board_readiness_history"]))
        story.append(Spacer(1, 14))

        # =================================================
        # KPI TABLE
        # =================================================
        if payload["primary_kpis"]:
            rows = [["Metric", "Value", "Confidence"]]
            row_styles = []

            for idx, kpi in enumerate(payload["primary_kpis"][:5], start=1):
                conf = kpi.get("confidence", 0.6)
                rows.append([
                    kpi.get("name"),
                    format_value(kpi.get("name", ""), kpi.get("value")),
                    f"{int(conf*100)}%",
                ])
                row_styles.append((
                    "BACKGROUND",
                    (0, idx),
                    (-1, idx),
                    HexColor("#dcfce7") if conf >= 0.85 else HexColor("#fef9c3") if conf >= 0.7 else HexColor("#fee2e2"),
                ))

            table = Table(rows, colWidths=[3.5*inch, 2*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
                ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (1,1), (-1,-1), "RIGHT"),
                ("PADDING", (0,0), (-1,-1), 8),
                *row_styles,
            ]))

            story.append(Paragraph("Key Performance Indicators", styles["ExecSection"]))
            story.append(table)

        # =================================================
        # VISUALS
        # =================================================
        if payload["visuals"]:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))

            for vis in payload["visuals"][:6]:
                p = Path(vis.get("path",""))
                if not p.exists():
                    continue
                try:
                    reader = utils.ImageReader(str(p))
                    iw, ih = reader.getSize()
                    aspect = ih / float(iw)
                    w = 6 * inch
                    h = min(w * aspect, 4 * inch)
                    story.append(Image(str(p), width=w, height=h))
                    story.append(Paragraph(vis.get("caption",""), styles["ExecCaption"]))
                    story.append(Spacer(1, 12))
                except Exception:
                    continue

        # =================================================
        # INSIGHTS
        # =================================================
        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))

        for ins in payload["insights"]:
            level = ins.get("level","INFO")
            color = "#dc2626" if level=="CRITICAL" else "#ea580c" if level=="RISK" else "#1f2937"
            story.append(Paragraph(
                f"<font color='{color}'><b>{level}</b></font> — {ins.get('title','')}",
                styles["ExecBody"]
            ))
            story.append(Paragraph(ins.get("so_what",""), styles["ExecBody"]))
            story.append(Spacer(1, 10))

        # =================================================
        # RECOMMENDATIONS
        # =================================================
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", styles["ExecSection"]))

        for rec in payload["recommendations"]:
            story.append(Paragraph(
                f"<b>{rec.get('priority','HIGH')}</b>: {rec.get('action','')}",
                styles["ExecBody"]
            ))
            if rec.get("timeline"):
                story.append(Paragraph(f"<i>Timeline:</i> {rec['timeline']}", styles["ExecBody"]))
            story.append(Spacer(1, 10))

        # =================================================
        # BUILD
        # =================================================
        doc.build(story)
        return output_path
