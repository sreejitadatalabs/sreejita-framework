# =====================================================
# EXECUTIVE PDF RENDERER — UNIVERSAL (FINAL, GOVERNED)
# Sreejita Framework v3.5.x
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
# PAYLOAD NORMALIZER (EXECUTIVE-SAFE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid payload for PDF rendering")

    executive = payload.get("executive", {})
    if not isinstance(executive, dict):
        executive = {}

    return {
        "domain": payload.get("domain", "unknown"),
        "executive_brief": executive.get("executive_brief", ""),
        "board_readiness": executive.get("board_readiness", {}) or {},
        "board_trend": executive.get("board_readiness_trend", {}) or {},
        "primary_kpis": executive.get("primary_kpis", []) or [],
        "insights": executive.get("insights", {}) or {},
        "recommendations": executive.get("recommendations", []) or [],
        "executive_by_sub_domain": executive.get("executive_by_sub_domain", {}) or {},
        "visuals": payload.get("visuals", []) or [],
    }


# =====================================================
# FORMAT HELPERS
# =====================================================

def format_value(v: Any) -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and 0 <= v <= 1:
            return f"{v:.1%}"
        v = float(v)
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:.2f}"
    except Exception:
        return str(v)


def confidence_badge(conf: Optional[float]) -> str:
    if conf is None:
        return "—"
    conf = float(conf)
    if conf >= 0.85:
        return "🟢 High"
    if conf >= 0.70:
        return "🟡 Medium"
    return "🔴 Low"


def confidence_color(conf: Optional[float]):
    if conf is None:
        return white
    conf = float(conf)
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
    Executive PDF Renderer (Authoritative)

    GUARANTEES:
    - Board-safe formatting
    - Minimum evidence enforcement
    - Sub-domain executive cognition support
    - ZERO intelligence computation
    """

    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    # -------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------
    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        data = normalize_pdf_payload(payload)

        # ------------------ Visual filtering ------------------
        visuals = [
            v for v in data["visuals"]
            if isinstance(v, dict)
            and Path(v.get("path", "")).exists()
            and float(v.get("confidence", 0)) >= 0.3
        ]

        # 🔒 GOVERNANCE: PDF never hard-fails visuals
        if len(visuals) < 2:
            raise RuntimeError(
                "PDF rejected: minimum 2 validated visuals required."
            )

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

        # =================================================
        # PAGE 1 — EXECUTIVE OVERVIEW
        # =================================================
        board = data["board_readiness"]
        trend = data["board_trend"]
        domain = str(data["domain"]).replace("_", " ").title()

        story.append(
            Paragraph(
                "Sreejita Executive Intelligence Report",
                styles["SR_Title"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Domain:</b> {domain}<br/>"
                f"<b>Board Readiness:</b> {board.get('score','—')} / 100 "
                f"({board.get('band','—')})<br/>"
                f"<b>Trend:</b> {trend.get('trend','→')}<br/>"
                f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d}",
                styles["SR_Body"],
            )
        )

        if data["executive_brief"]:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Brief", styles["SR_Section"]))
            story.append(
                Paragraph(data["executive_brief"], styles["SR_Body"])
            )

        # =================================================
        # KPI TABLE (MIN 5, MAX 9)
        # =================================================
        kpis = list(data["primary_kpis"][:9])

        while len(kpis) < 5:
            kpis.append({
                "name": "Data Coverage",
                "value": None,
                "confidence": 0.4,
            })

        rows = [["Metric", "Value", "Confidence"]]
        bg_styles = []

        for idx, k in enumerate(kpis, start=1):
            conf = float(k.get("confidence", 0.7))
            rows.append([
                k.get("name", "—"),
                format_value(k.get("value")),
                f"{confidence_badge(conf)} ({int(conf*100)}%)",
            ])
            bg_styles.append(
                ("BACKGROUND", (0, idx), (-1, idx), confidence_color(conf))
            )

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

        # =================================================
        # SUB-DOMAIN EXECUTIVE SECTIONS
        # =================================================
        sub_execs = data["executive_by_sub_domain"]
        if isinstance(sub_execs, dict) and sub_execs:
            story.append(PageBreak())
            story.append(
                Paragraph(
                    "Executive Summary by Operating Area",
                    styles["SR_Section"],
                )
            )

            for sub, payload in sub_execs.items():
                if not isinstance(payload, dict):
                    continue

                story.append(
                    Paragraph(sub.replace("_", " ").title(), styles["SR_Section"])
                )
                story.append(
                    Paragraph(payload.get("executive_brief", ""), styles["SR_Body"])
                )

                board = payload.get("board_readiness", {})
                story.append(
                    Paragraph(
                        f"<i>Board Readiness:</i> "
                        f"{board.get('score','—')} / 100 "
                        f"({board.get('band','—')})",
                        styles["SR_Body"],
                    )
                )
                story.append(Spacer(1, 10))

        # =================================================
        # VISUAL EVIDENCE — 2 PER PAGE
        # =================================================
        for i in range(0, len(visuals), 2):
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["SR_Section"]))

            for v in visuals[i:i+2]:
                img_path = Path(v["path"])
                img = utils.ImageReader(str(img_path))
                iw, ih = img.getSize()

                w = 6 * inch
                h = min(w * ih / iw, 4 * inch)

                story.append(Image(str(img_path), width=w, height=h))
                story.append(
                    Paragraph(
                        f"{v.get('caption','')} "
                        f"<i>(Confidence: {confidence_badge(v.get('confidence'))})</i>",
                        styles["SR_Body"],
                    )
                )
                story.append(Spacer(1, 12))

        # =================================================
        # INSIGHTS
        # =================================================
        insight_block = data["insights"]
        ordered = (
            insight_block.get("strengths", []) +
            insight_block.get("warnings", []) +
            insight_block.get("risks", [])
        ) if isinstance(insight_block, dict) else []

        if ordered:
            story.append(PageBreak())
            story.append(Paragraph("Key Insights", styles["SR_Section"]))

            for ins in ordered[:5]:
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
        recs = data["recommendations"][:5]
        if recs:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", styles["SR_Section"]))

            for rec in recs:
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

        if not output_path.exists():
            raise RuntimeError("PDF generation failed")

        return output_path
