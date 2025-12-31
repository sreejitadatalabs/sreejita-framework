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
    payload.setdefault("scorecard", {})
    return payload


# =====================================================
# FORMATTERS
# =====================================================

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
        story: List[Any] = []

        # -------------------------------------------------
        # SAFE CUSTOM STYLES (NO NAME COLLISION)
        # -------------------------------------------------
        styles.add(ParagraphStyle(
            name="ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=18,
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

        # =================================================
        # COVER PAGE (BRANDED)
        # =================================================
        story.append(Paragraph(
            "SREEJITA INTELLIGENCE FRAMEWORK™",
            styles["ExecTitle"]
        ))

        story.append(Paragraph(
            "Executive Healthcare Performance Report",
            styles["ExecSection"]
        ))

        story.append(Paragraph(
            f"Domain: Healthcare Operations<br/>"
            f"Confidence Level: {payload['scorecard'].get('risk_label', 'N/A')}<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["ExecBody"]
        ))

        story.append(Spacer(1, 12))

        story.append(Paragraph(
            "Prepared by: <b>Sreejita Data Labs</b>",
            styles["ExecBody"]
        ))

        story.append(PageBreak())

        # =================================================
        # EXECUTIVE DECISION SNAPSHOT (PAGE 1)
        # =================================================
        snapshot = payload.get("executive_snapshot")
        if snapshot:
            story.append(Paragraph(snapshot.get("title", "EXECUTIVE DECISION SNAPSHOT"),
                                   styles["ExecSection"]))

            story.append(Paragraph(
                f"<b>Overall Risk:</b> {snapshot.get('overall_risk', '-')}",
                styles["ExecBody"]
            ))

            story.append(Spacer(1, 8))

            story.append(Paragraph("<b>Top Problems:</b>", styles["ExecBody"]))
            for p in snapshot.get("top_problems", []):
                story.append(Paragraph(f"• {p}", styles["ExecBody"]))

            story.append(Spacer(1, 8))

            story.append(Paragraph("<b>Decisions Required:</b>", styles["ExecBody"]))
            for d in snapshot.get("decisions_required", []):
                story.append(Paragraph(f"☐ {d}", styles["ExecBody"]))

            story.append(Spacer(1, 10))

            story.append(Paragraph(
                "<b>Confidence Scale:</b> "
                "85–100 = 🟢 Green | 70–84 = 🟡 Yellow | "
                "50–69 = 🟠 Orange | <50 = 🔴 Red",
                styles["ExecCaption"]
            ))

            story.append(PageBreak())

        # =================================================
        # EXECUTIVE SUMMARY
        # =================================================
        if payload["summary"]:
            story.append(Paragraph("Executive Summary", styles["ExecSection"]))
            for s in payload["summary"]:
                story.append(Paragraph(f"• {s}", styles["ExecBody"]))
            story.append(PageBreak())

        # =================================================
        # PRIMARY KPIs (TOP 3–5)
        # =================================================
        if payload["primary_kpis"]:
            story.append(Paragraph("Key Performance Indicators", styles["ExecSection"]))

            table_data = [["Metric", "Value"]]
            for item in payload["primary_kpis"][:5]:
                table_data.append([
                    item.get("name", "-"),
                    format_kpi_value(item.get("name", ""), item.get("value"))
                ])

            table = Table(table_data, colWidths=[4 * inch, 2 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))

            story.append(table)
            story.append(PageBreak())

        # =================================================
        # VISUAL EVIDENCE (MAX 6)
        # =================================================
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

        # =================================================
        # INSIGHTS (TOP 3–5 ONLY)
        # =================================================
        if payload["insights"]:
            story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))

            for i in payload["insights"][:5]:
                story.append(Paragraph(
                    f"<b>{i.get('level','INFO')}:</b> {i.get('title','')}",
                    styles["ExecBody"]
                ))
                story.append(Paragraph(i.get("so_what", ""), styles["ExecBody"]))
                story.append(Spacer(1, 8))

            story.append(PageBreak())

        # =================================================
        # RECOMMENDATIONS (DECISION FRAMED)
        # =================================================
        if payload["recommendations"]:
            story.append(Paragraph("Recommendations", styles["ExecSection"]))
            story.append(Paragraph(
                "<b>The following actions require executive approval to mitigate identified risks.</b>",
                styles["ExecBody"]
            ))

            for idx, r in enumerate(payload["recommendations"][:5], start=1):
                story.append(Paragraph(
                    f"{idx}. {r.get('action','Action required')}",
                    styles["ExecBody"]
                ))

                meta = []
                if r.get("timeline"):
                    meta.append(f"Timeline: {r['timeline']}")
                if r.get("owner"):
                    meta.append(f"Owner: {r['owner']}")
                if r.get("expected_outcome"):
                    meta.append(f"Success: {r['expected_outcome']}")

                if meta:
                    story.append(Paragraph(" | ".join(meta), styles["ExecCaption"]))

                story.append(Spacer(1, 10))

        # =================================================
        # BUILD DOCUMENT
        # =================================================
        doc.build(story)
        return output_path
