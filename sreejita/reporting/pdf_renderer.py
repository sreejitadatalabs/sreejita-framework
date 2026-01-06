# =====================================================
# EXECUTIVE PDF RENDERER — UNIVERSAL (FINAL, HARDENED)
# Sreejita Framework v3.6
# =====================================================

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
)
from reportlab.lib.units import inch
from reportlab.lib import utils


# =====================================================
# SAFE NORMALIZATION
# =====================================================

def _safe_float(v, default=0.0):
    try:
        v = float(v)
        return max(0.0, min(v, 1.0))
    except Exception:
        return default


def format_value(v: Any) -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, (int, float)):
            if 0 <= v <= 1:
                return f"{v:.1%}"
            if abs(v) >= 1_000_000:
                return f"{v/1_000_000:.1f}M"
            if abs(v) >= 1_000:
                return f"{v/1_000:.1f}K"
            return f"{v:.2f}"
        return str(v)
    except Exception:
        return "—"


def confidence_badge(conf: float) -> str:
    if conf >= 0.85:
        return "High"
    if conf >= 0.70:
        return "Medium"
    return "Low"


def confidence_color(conf: float):
    if conf >= 0.85:
        return HexColor("#dcfce7")
    if conf >= 0.70:
        return HexColor("#fef9c3")
    return HexColor("#fee2e2")


# =====================================================
# EXECUTIVE PDF RENDERER
# =====================================================

class ExecutivePDFRenderer:
    """
    PDF Renderer — strictly PRESENTATIONAL

    GUARANTEES:
    - Never crashes on weak data
    - Never rejects reports
    - Never computes intelligence
    """

    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    # -------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------
    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid payload")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        story: List[Any] = []

        # -------------------------------------------------
        # STYLES
        # -------------------------------------------------
        styles.add(ParagraphStyle(
            "SR_Title",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=18,
            fontName="Helvetica-Bold",
        ))

        styles.add(ParagraphStyle(
            "SR_Section",
            fontSize=15,
            spaceBefore=18,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        ))

        styles.add(ParagraphStyle(
            "SR_Body",
            fontSize=11,
            leading=15,
            spaceAfter=6,
        ))

        executive = payload.get("executive", {}) or {}
        visuals = payload.get("visuals", []) or []
        insights = payload.get("insights", []) or []
        recommendations = payload.get("recommendations", []) or []

        # =================================================
        # PAGE 1 — EXECUTIVE OVERVIEW
        # =================================================
        board = executive.get("board_readiness", {}) or {}
        trend = executive.get("board_readiness_trend", {}) or {}

        story.append(
            Paragraph(
                "Sreejita Executive Intelligence Report",
                styles["SR_Title"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Domain:</b> {payload.get('domain','—').title()}<br/>"
                f"<b>Board Readiness:</b> {board.get('score','—')} / 100 "
                f"({board.get('band','—')})<br/>"
                f"<b>Trend:</b> {trend.get('trend','→')}<br/>"
                f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d}",
                styles["SR_Body"],
            )
        )

        brief = executive.get("executive_brief")
        if isinstance(brief, str) and brief.strip():
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Brief", styles["SR_Section"]))
            story.append(Paragraph(brief, styles["SR_Body"]))

        # =================================================
        # KPI TABLE (SAFE, FROM RAW KPIs)
        # =================================================
        raw_kpis = payload.get("kpis", {}) or {}
        kpi_items = [
            (k, v) for k, v in raw_kpis.items()
            if isinstance(k, str) and not k.startswith("_")
        ][:9]

        while len(kpi_items) < 5:
            kpi_items.append(("Data Coverage", None))

        rows = [["Metric", "Value"]]
        for k, v in kpi_items:
            rows.append([k.replace("_", " ").title(), format_value(v)])

        table = Table(rows, colWidths=[4.5 * inch, 2.5 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))

        story.append(Spacer(1, 14))
        story.append(Paragraph("Key Performance Indicators", styles["SR_Section"]))
        story.append(table)

        # =================================================
        # VISUAL EVIDENCE (NO HARD FAIL)
        # =================================================
        valid_visuals = [
            v for v in visuals
            if isinstance(v, dict) and Path(v.get("path", "")).exists()
        ][:6]

        if valid_visuals:
            for i in range(0, len(valid_visuals), 2):
                story.append(PageBreak())
                story.append(Paragraph("Visual Evidence", styles["SR_Section"]))

                for v in valid_visuals[i:i + 2]:
                    img_path = Path(v["path"])
                    img = utils.ImageReader(str(img_path))
                    iw, ih = img.getSize()

                    w = 6 * inch
                    h = min(w * ih / iw, 4 * inch)

                    conf = _safe_float(v.get("confidence"))

                    story.append(Image(str(img_path), width=w, height=h))
                    story.append(
                        Paragraph(
                            f"{v.get('caption','')} "
                            f"<i>(Confidence: {confidence_badge(conf)})</i>",
                            styles["SR_Body"],
                        )
                    )
                    story.append(Spacer(1, 12))

        # =================================================
        # INSIGHTS
        # =================================================
        if insights:
            story.append(PageBreak())
            story.append(Paragraph("Key Insights", styles["SR_Section"]))

            for ins in insights[:5]:
                if not isinstance(ins, dict):
                    continue
                story.append(
                    Paragraph(
                        f"<b>{ins.get('level','INFO')}:</b> {ins.get('title','')}",
                        styles["SR_Body"],
                    )
                )
                story.append(
                    Paragraph(ins.get("so_what",""), styles["SR_Body"])
                )
                story.append(Spacer(1, 8))

        # =================================================
        # RECOMMENDATIONS
        # =================================================
        if recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", styles["SR_Section"]))

            for rec in recommendations[:5]:
                if not isinstance(rec, dict):
                    continue
                story.append(
                    Paragraph(
                        f"<b>{rec.get('priority','')}:</b> {rec.get('action','')}",
                        styles["SR_Body"],
                    )
                )
                story.append(
                    Paragraph(
                        f"<i>Owner:</i> {rec.get('owner','—')} | "
                        f"<i>Timeline:</i> {rec.get('timeline','—')}<br/>"
                        f"<i>Goal:</i> {rec.get('goal','—')}",
                        styles["SR_Body"],
                    )
                )
                story.append(Spacer(1, 10))

        doc.build(story)

        return output_path
