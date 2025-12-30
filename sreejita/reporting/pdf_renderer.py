# sreejita/reporting/pdf_renderer.py

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
# PAYLOAD NORMALIZER (UNIVERSAL, DEFENSIVE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    payload.setdefault("meta", {})
    payload.setdefault("executive_snapshot", None)
    payload.setdefault("primary_kpis", [])
    payload.setdefault("scorecard", {})
    payload.setdefault("summary", payload.get("executive_summary", []))
    payload.setdefault("visuals", [])
    payload.setdefault("insights", [])
    payload.setdefault("recommendations", [])
    return payload


# =====================================================
# FORMATTERS
# =====================================================

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
    MUTED = HexColor("#6b7280")

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

        # ------------------------------
        # STYLES
        # ------------------------------
        styles.add(ParagraphStyle(
            name="Title",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName="Helvetica-Bold",
            textColor=self.PRIMARY
        ))
        styles.add(ParagraphStyle(
            name="Section",
            fontSize=15,
            spaceBefore=18,
            spaceAfter=10,
            fontName="Helvetica-Bold"
        ))
        styles.add(ParagraphStyle(
            name="Body",
            fontSize=11,
            leading=15,
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name="Caption",
            fontSize=9,
            alignment=TA_CENTER,
            textColor=self.MUTED,
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name="Footer",
            fontSize=8,
            alignment=TA_CENTER,
            textColor=self.MUTED
        ))

        # =================================================
        # COVER
        # =================================================
        story.append(Paragraph("Sreejita Executive Intelligence Report", styles["Title"]))
        story.append(Paragraph(
            f"Generated on {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["Body"]
        ))
        story.append(Spacer(1, 16))

        # =================================================
        # EXECUTIVE DECISION SNAPSHOT
        # =================================================
        snap = payload.get("executive_snapshot")
        if isinstance(snap, dict):
            story.append(Paragraph("Executive Decision Snapshot", styles["Section"]))

            risk = snap.get("overall_risk", {})
            if risk:
                story.append(Paragraph(
                    f"<b>Overall Risk:</b> {risk.get('icon','')} {risk.get('label','')} "
                    f"(Score: {risk.get('score','-')})",
                    styles["Body"]
                ))
                story.append(Spacer(1, 6))

            for p in snap.get("top_problems", []):
                story.append(Paragraph(f"• {p}", styles["Body"]))

            story.append(Spacer(1, 6))
            for a in snap.get("top_actions", []):
                story.append(Paragraph(f"→ {a}", styles["Body"]))

            if snap.get("decisions_required"):
                story.append(Spacer(1, 8))
                for d in snap["decisions_required"]:
                    story.append(Paragraph(f"☐ {d}", styles["Body"]))

            story.append(PageBreak())

        # =================================================
        # PRIMARY KPIs (MAX 5)
        # =================================================
        pkpis = payload.get("primary_kpis", [])
        if pkpis:
            story.append(Paragraph("Key Performance Indicators", styles["Section"]))

            table_data = [["Metric", "Value"]]
            for item in pkpis[:5]:
                table_data.append([
                    item.get("name", "-"),
                    fmt(item.get("value"))
                ])

            table = Table(table_data, colWidths=[4.5 * inch, 1.8 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
                ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (1,1), (-1,-1), "RIGHT"),
                ("PADDING", (0,0), (-1,-1), 8),
            ]))

            story.append(table)
            story.append(PageBreak())

        # =================================================
        # EXECUTIVE SUMMARY
        # =================================================
        if payload.get("summary"):
            story.append(Paragraph("Executive Summary", styles["Section"]))
            for s in payload["summary"]:
                story.append(Paragraph(f"• {s}", styles["Body"]))
            story.append(PageBreak())

        # =================================================
        # VISUAL EVIDENCE (MAX 6)
        # =================================================
        visuals = sorted(
            payload.get("visuals", []),
            key=lambda x: x.get("importance", 0),
            reverse=True
        )

        if visuals:
            story.append(Paragraph("Visual Evidence", styles["Section"]))

            for vis in visuals[:6]:
                path = Path(vis.get("path", ""))
                if path.exists():
                    img = utils.ImageReader(str(path))
                    iw, ih = img.getSize()
                    width = 6 * inch
                    height = min((ih / iw) * width, 5 * inch)
                    story.append(Image(str(path), width=width, height=height))
                    if vis.get("caption"):
                        story.append(Paragraph(vis["caption"], styles["Caption"]))

            story.append(PageBreak())

        # =================================================
        # INSIGHTS & RISKS
        # =================================================
        if payload.get("insights"):
            story.append(Paragraph("Key Insights & Risks", styles["Section"]))
            for i in payload["insights"]:
                story.append(Paragraph(
                    f"<b>{i.get('level','INFO')}:</b> {i.get('title','')}",
                    styles["Body"]
                ))
                story.append(Paragraph(i.get("so_what",""), styles["Body"]))
                story.append(Spacer(1, 8))
            story.append(PageBreak())

        # =================================================
        # RECOMMENDATIONS (MAX 5)
        # =================================================
        if payload.get("recommendations"):
            story.append(Paragraph("Recommendations", styles["Section"]))

            for idx, r in enumerate(payload["recommendations"][:5], start=1):
                story.append(Paragraph(
                    f"{idx}. {r.get('action','Action required')}",
                    styles["Body"]
                ))

                meta = []
                if r.get("timeline"):
                    meta.append(f"Timeline: {r['timeline']}")
                if r.get("owner"):
                    meta.append(f"Owner: {r['owner']}")
                if r.get("expected_outcome"):
                    meta.append(f"Success: {r['expected_outcome']}")

                if meta:
                    story.append(Paragraph(" | ".join(meta), styles["Caption"]))

                story.append(Spacer(1, 10))

        # =================================================
        # FOOTER
        # =================================================
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "Generated by Sreejita Framework • Executive Intelligence Engine",
            styles["Footer"]
        ))

        doc.build(story)
        return output_path
