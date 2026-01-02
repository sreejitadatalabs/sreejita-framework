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
# PAYLOAD NORMALIZER (AUTHORITATIVE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single source of truth for PDF rendering.
    Accepts:
    - orchestrator payload
    - executive cognition payload
    - domain-wrapped payloads
    """

    if not isinstance(payload, dict):
        return {}

    # unwrap nested executive blocks safely
    if "executive" in payload:
        executive = payload["executive"]
    elif "executive_brief" in payload:
        executive = payload
    elif len(payload) == 1:
        executive = next(iter(payload.values()), {})
    else:
        executive = {}

    return {
        "executive_brief": executive.get("executive_brief", ""),
        "primary_kpis": executive.get("primary_kpis", []),
        "board_readiness": executive.get("board_readiness", {}),
        "insights": executive.get("insights", {}),
        "recommendations": executive.get("recommendations", []),
        "visuals": payload.get("visuals", []),
        "sub_domain": executive.get("sub_domain", "unknown"),
    }


# =====================================================
# FORMAT HELPERS
# =====================================================

def format_value(v: Any) -> str:
    if v is None:
        return "-"
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
# EXECUTIVE PDF RENDERER
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
        # SAFE CUSTOM STYLES
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

        # =====================================================
        # PAGE 1 — EXECUTIVE SUMMARY
        # =====================================================

        br = payload.get("board_readiness", {})
        sub_domain = payload.get("sub_domain", "healthcare").replace("_", " ").title()

        story.append(Paragraph("Sreejita Executive Intelligence Report", styles["SR_Title"]))
        story.append(Paragraph(
            f"<b>Domain:</b> {sub_domain}<br/>"
            f"<b>Board Readiness:</b> {br.get('score','-')} / 100 "
            f"({br.get('band','-')})<br/>"
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d}",
            styles["SR_Body"],
        ))

        # Executive Brief
        if payload.get("executive_brief"):
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Brief", styles["SR_Section"]))
            story.append(Paragraph(payload["executive_brief"], styles["SR_Body"]))

        # =====================================================
        # KPI TABLE (MAX 9)
        # =====================================================

        kpis = payload.get("primary_kpis", [])
        if kpis:
            rows = [["Metric", "Value", "Confidence"]]
            bg_styles = []

            for idx, k in enumerate(kpis[:9], start=1):
                conf = k.get("confidence", 0.7)
                rows.append([
                    k.get("name", "Unknown"),
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

        # =====================================================
        # VISUAL EVIDENCE (FILTERED)
        # =====================================================

        visuals = [
            v for v in payload.get("visuals", [])
            if v.get("confidence", 0) >= 0.3
            and Path(v.get("path", "")).exists()
        ]

        story.append(PageBreak())
        story.append(Paragraph("Visual Evidence", styles["SR_Section"]))

        if not visuals:
            story.append(Paragraph(
                "No statistically reliable visuals could be generated from this dataset.",
                styles["SR_Body"],
            ))
        else:
            for i in range(0, len(visuals), 2):
                if i > 0:
                    story.append(PageBreak())
                    story.append(Paragraph("Visual Evidence (Continued)", styles["SR_Section"]))

                for v in visuals[i:i+2]:
                    img_path = Path(v["path"])
                    img = utils.ImageReader(str(img_path))
                    iw, ih = img.getSize()

                    w = 6 * inch
                    h = min(w * ih / iw, 4 * inch)

                    story.append(Image(str(img_path), width=w, height=h))
                    story.append(Paragraph(
                        f"{v.get('caption','')} "
                        f"<i>(Confidence: {confidence_badge(v.get('confidence'))})</i>",
                        styles["SR_Body"],
                    ))
                    story.append(Spacer(1, 12))

        # =====================================================
        # INSIGHTS
        # =====================================================

        insight_block = payload.get("insights", {})
        if insight_block:
            story.append(PageBreak())
            story.append(Paragraph("Key Insights", styles["SR_Section"]))

            for group in ["strengths", "warnings", "risks"]:
                for ins in insight_block.get(group, []):
                    story.append(Paragraph(
                        f"<b>{ins.get('level','INFO')}:</b> {ins.get('title','')}",
                        styles["SR_Body"],
                    ))
                    story.append(Paragraph(ins.get("so_what",""), styles["SR_Body"]))
                    story.append(Spacer(1, 8))

        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        recs = payload.get("recommendations", [])
        if recs:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", styles["SR_Section"]))

            for rec in recs[:7]:
                story.append(Paragraph(
                    f"<b>{rec.get('priority','')}:</b> {rec.get('action','')}",
                    styles["SR_Body"],
                ))
                story.append(Paragraph(
                    f"<i>Owner:</i> {rec.get('owner','-')} | "
                    f"<i>Timeline:</i> {rec.get('timeline','-')}<br/>"
                    f"<i>Goal:</i> {rec.get('goal','-')}",
                    styles["SR_Body"],
                ))
                story.append(Spacer(1, 10))

        doc.build(story)

        if not output_path.exists():
            raise RuntimeError("PDF generation failed")

        return output_path
