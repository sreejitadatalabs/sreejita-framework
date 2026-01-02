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
# PAYLOAD NORMALIZER (SAFE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    # unwrap domain key if present
    if len(payload) == 1 and "executive" not in payload:
        payload = next(iter(payload.values()), {})

    executive = payload.get("executive", {}) or {}

    return {
        "snapshot": executive.get("snapshot", {}),
        "primary_kpis": executive.get("primary_kpis", []),
        "board_readiness": executive.get("board_readiness", {}),
        "board_readiness_trend": executive.get("board_readiness_trend", {}),
        "board_readiness_history": executive.get("board_readiness_history", []),
        "insights": payload.get("insights", []),
        "recommendations": payload.get("recommendations", []),
        "visuals": payload.get("visuals", []),
    }


# =====================================================
# FORMAT HELPERS
# =====================================================

def format_value(v: Any) -> str:
    if v is None:
        return "-"
    try:
        if isinstance(v, float) and v <= 1:
            return f"{v:.1%}"
        if abs(float(v)) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(float(v)) >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


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
# BOARD READINESS SPARKLINE
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
# EXECUTIVE PDF RENDERER (STABLE)
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
    
        # ---------- SAFE CUSTOM STYLES ----------
        if "SR_Title" not in styles:
            styles.add(ParagraphStyle(
                "SR_Title", fontSize=22, alignment=TA_CENTER,
                spaceAfter=18, fontName="Helvetica-Bold"
            ))
        if "SR_Section" not in styles:
            styles.add(ParagraphStyle(
                "SR_Section", fontSize=15,
                spaceBefore=18, spaceAfter=10,
                fontName="Helvetica-Bold"
            ))
        if "SR_Body" not in styles:
            styles.add(ParagraphStyle(
                "SR_Body", fontSize=11, leading=15, spaceAfter=6
            ))
    
        # =====================================================
        # PAGE 1 — EXECUTIVE BRIEF + KPIs
        # =====================================================
        br = payload["board_readiness"]
    
        story.append(Paragraph("Sreejita Executive Report", styles["SR_Title"]))
        story.append(Paragraph(
            f"<b>Board Readiness:</b> {br.get('score','-')} / 100 "
            f"({br.get('band','-' )})<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["SR_Body"],
        ))
    
        # Executive Brief
        snapshot = payload.get("snapshot", {})
        if snapshot:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Brief", styles["SR_Section"]))
            story.append(Paragraph(
                snapshot.get("overall_risk", ""),
                styles["SR_Body"],
            ))
    
        # KPI TABLE
        if payload["primary_kpis"]:
            rows = [["Metric", "Value", "Confidence"]]
            bg_styles = []
    
            for idx, k in enumerate(payload["primary_kpis"][:5], start=1):
                conf = k.get("confidence")
                rows.append([
                    k.get("name"),
                    format_value(k.get("value")),
                    f"{confidence_badge(conf)} ({int(conf*100)}%)",
                ])
                bg_styles.append((
                    "BACKGROUND", (0, idx), (-1, idx), confidence_color(conf)
                ))
    
            table = Table(rows, colWidths=[3.5*inch, 2*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
                ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("PADDING", (0,0), (-1,-1), 8),
                *bg_styles,
            ]))
    
            story.append(Spacer(1, 14))
            story.append(Paragraph("Key Performance Indicators", styles["SR_Section"]))
            story.append(table)
    
        # =====================================================
        # PAGE 2+ — VISUAL EVIDENCE (2 PER PAGE)
        # =====================================================
        visuals = payload.get("visuals", [])
        visual_pages = [visuals[i:i+2] for i in range(0, len(visuals), 2)]
    
        for page_idx, pair in enumerate(visual_pages):
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["SR_Section"]))
    
            for v in pair:
                p = Path(v.get("path", ""))
                if p.exists():
                    img = utils.ImageReader(str(p))
                    iw, ih = img.getSize()
                    w = 6 * inch
                    h = min(w * ih / iw, 4 * inch)
                    story.append(Image(str(p), width=w, height=h))
                    if v.get("caption"):
                        story.append(Paragraph(v["caption"], styles["SR_Body"]))
                    story.append(Spacer(1, 12))
    
        # =====================================================
        # INSIGHTS (AFTER VISUALS)
        # =====================================================
        if payload["insights"]:
            story.append(PageBreak())
            story.append(Paragraph("Key Insights", styles["SR_Section"]))
    
            for ins in payload["insights"][:5]:
                story.append(Paragraph(
                    f"<b>{ins.get('level','INFO')}:</b> {ins.get('title','')}",
                    styles["SR_Body"],
                ))
                story.append(Paragraph(ins.get("so_what",""), styles["SR_Body"]))
                story.append(Spacer(1, 8))
    
        # =====================================================
        # RECOMMENDATIONS (FLOW-BASED)
        # =====================================================
        if payload["recommendations"]:
            story.append(Spacer(1, 14))
            story.append(Paragraph("Recommendations", styles["SR_Section"]))
    
            for rec in payload["recommendations"][:5]:
                story.append(Paragraph(
                    f"<b>{rec.get('priority','')}:</b> {rec.get('action','')}",
                    styles["SR_Body"],
                ))
                if rec.get("timeline"):
                    story.append(Paragraph(
                        f"<i>Timeline:</i> {rec['timeline']}",
                        styles["SR_Body"],
                    ))
                story.append(Spacer(1, 8))
    
        doc.build(story)
    
        if not output_path.exists():
            raise RuntimeError("PDF generation failed")
    
        return output_path
