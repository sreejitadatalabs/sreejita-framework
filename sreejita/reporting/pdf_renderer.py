from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
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
# PAYLOAD NORMALIZER
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    payload.setdefault("meta", {})
    payload.setdefault("executive_snapshot", None)
    payload.setdefault("primary_kpis", [])
    payload.setdefault("summary", [])
    payload.setdefault("visuals", [])
    payload.setdefault("insights", [])
    payload.setdefault("recommendations", [])
    return payload

def format_kpi_value(key, value):
    if value is None:
        return "-"

    if isinstance(value, (int, float)):
        k = key.lower()

        if "rate" in k or "ratio" in k:
            return f"{value:.1%}"

        if "los" in k or "days" in k:
            return f"{value:.1f} days"

        if "cost" in k or "billing" in k:
            if value >= 1_000_000:
                return f"${value/1_000_000:.1f}M"
            if value >= 1_000:
                return f"${value/1_000:.1f}K"
            return f"${value:,.0f}"

        return f"{value:,.0f}"

    return str(value)
    
def fmt(val):
    if val is None or val == "":
        return "-"
    try:
        if isinstance(val, float) and pd.isna(val):
            return "-"
    except Exception:
        pass
    return str(val)


# =====================================================
# EXECUTIVE PDF RENDERER
# =====================================================

class ExecutivePDFRenderer:
    PRIMARY = HexColor("#1f2937")
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
        story = []

        # ✅ SAFE CUSTOM STYLES (NO COLLISION)
        styles.add(ParagraphStyle(
            name="ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName="Helvetica-Bold",
            textColor=self.PRIMARY
        ))
        styles.add(ParagraphStyle(
            name="ExecSection",
            fontSize=15,
            spaceBefore=18,
            spaceAfter=10,
            fontName="Helvetica-Bold"
        ))
        styles.add(ParagraphStyle(
            name="ExecBody",
            fontSize=11,
            leading=15,
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name="ExecCaption",
            fontSize=9,
            alignment=TA_CENTER,
            textColor=HexColor("#6b7280"),
            spaceAfter=12
        ))

        # -------------------------------------------------
        # COVER
        # -------------------------------------------------
        story.append(Paragraph(
            "SREEJITA INTELLIGENCE FRAMEWORK™",
            styles["Title"]
        ))
    
        story.append(Paragraph(
                "Executive Healthcare Performance Report",
                styles["Section"]
            ))
        
        story.append(Paragraph(
            f"Domain: Healthcare Operations<br/>"
            f"Confidence Level: {payload.get('scorecard', {}).get('risk_label', 'N/A')}<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["Body"]
        ))
        
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(
            "Prepared by: <b>Sreejita Data Labs</b>",
            styles["Body"]
        ))
        
        story.append(PageBreak())
    
        # -------------------------------------------------
        # EXECUTIVE SUMMARY
        # -------------------------------------------------
        if payload["summary"]:
            story.append(Paragraph("Executive Summary", styles["ExecSection"]))
            for s in payload["summary"]:
                story.append(Paragraph(f"• {s}", styles["ExecBody"]))
            story.append(PageBreak())

        # -------------------------------------------------
        # PRIMARY KPIs (MAX 5)
        # -------------------------------------------------
        if payload["primary_kpis"]:
            story.append(Paragraph("Key Performance Indicators", styles["ExecSection"]))
            table_data = [["Metric", "Value"]]
            for item in payload["primary_kpis"][:5]:
                table_data.append([item.get("name", "-"), format_kpi_value(item.get("name", ""), item.get("value"))

            t = Table(table_data, colWidths=[4 * inch, 2 * inch])
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(t)
            story.append(PageBreak())

        # -------------------------------------------------
        # VISUALS (MAX 6)
        # -------------------------------------------------
        if payload["visuals"]:
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))
            for vis in payload["visuals"][:6]:
                path = Path(vis.get("path", ""))
                if path.exists():
                    img = utils.ImageReader(str(path))
                    iw, ih = img.getSize()
                    w = 6 * inch
                    h = min((ih / iw) * w, 5 * inch)
                    story.append(Image(str(path), width=w, height=h))
                    story.append(Paragraph(vis.get("caption", ""), styles["ExecCaption"]))
            story.append(PageBreak())

        # -------------------------------------------------
        # INSIGHTS
        # -------------------------------------------------
        if payload["insights"]:
            story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))
            for i in payload["insights"]:
                story.append(Paragraph(
                    f"<b>{i.get('level','INFO')}:</b> {i.get('title','')}",
                    styles["ExecBody"]
                ))
                story.append(Paragraph(i.get("so_what", ""), styles["ExecBody"]))
                story.append(Spacer(1, 8))

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------
        if payload["recommendations"]:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", styles["ExecSection"]))
            for idx, r in enumerate(payload["recommendations"][:5], start=1):
                story.append(Paragraph(
                    f"{idx}. {r.get('action','Action required')}",
                    styles["ExecBody"]
                ))
                meta = []
                if r.get("timeline"): meta.append(f"Timeline: {r['timeline']}")
                if r.get("owner"): meta.append(f"Owner: {r['owner']}")
                if r.get("expected_outcome"):
                    meta.append(f"Success: {r['expected_outcome']}")
                if meta:
                    story.append(Paragraph(" | ".join(meta), styles["ExecCaption"]))
                story.append(Spacer(1, 10))

        doc.build(story)
        return output_path
