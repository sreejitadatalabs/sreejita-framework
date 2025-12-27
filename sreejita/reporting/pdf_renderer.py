# sreejita/reporting/pdf_renderer.py
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import tempfile
import os

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
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


# =====================================================
# KPI FORMATTERS
# =====================================================

def format_number(x):
    try:
        x = float(x)
    except Exception:
        return str(x)

    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    if abs(x) < 1 and x != 0:
        return f"{x:.2f}"
    return f"{int(x):,}"


def format_percent(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


# =====================================================
# EXECUTIVE PDF RENDERER (STABLE)
# =====================================================

class ExecutivePDFRenderer:
    """
    Sreejita v3.5.1 — Executive PDF Renderer (ReportLab)

    ✔ Streamlit-safe
    ✔ GitHub Web-safe
    ✔ No browser
    ✔ Deterministic
    """

    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(
        self,
        payload: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        output_path = Path(output_path)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        story: List = []

        # -------------------------
        # STYLES
        # -------------------------
        styles.add(ParagraphStyle(
            name="Title",
            fontSize=22,
            spaceAfter=24,
            alignment=TA_CENTER,
            textColor=self.PRIMARY,
        ))

        styles.add(ParagraphStyle(
            name="Section",
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=self.PRIMARY,
        ))

        styles.add(ParagraphStyle(
            name="Body",
            fontSize=11,
            leading=14,
        ))

        # -------------------------
        # COVER
        # -------------------------
        story.append(Paragraph("Sreejita Executive Report", styles["Title"]))
        story.append(Spacer(1, 12))

        meta = payload.get("meta", {})
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["Body"],
        ))

        story.append(PageBreak())

        # -------------------------
        # EXECUTIVE SUMMARY
        # -------------------------
        story.append(Paragraph("Executive Summary", styles["Section"]))
        for item in payload.get("summary", []):
            story.append(Paragraph(f"• {item}", styles["Body"]))
            story.append(Spacer(1, 6))

        # -------------------------
        # KPI SNAPSHOT
        # -------------------------
        story.append(Paragraph("Key Metrics", styles["Section"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            if any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion"]):
                value = format_percent(v)
            else:
                value = format_number(v)
            table_data.append([k.replace("_", " ").title(), value])

        table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(table)

        # -------------------------
        # VISUAL EVIDENCE
        # -------------------------
        temp_images: List[str] = []

        visuals = payload.get("visuals", [])
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["Section"]))

            for vis in visuals:
                img_path = self._render_chart(vis)
                temp_images.append(img_path)

                story.append(Image(
                    img_path,
                    width=5.5 * inch,
                    height=3.2 * inch,
                ))
                story.append(Paragraph(vis.get("caption", ""), styles["Body"]))
                story.append(Spacer(1, 12))

        # -------------------------
        # INSIGHTS
        # -------------------------
        story.append(PageBreak())
        story.append(Paragraph("Insights & Risks", styles["Section"]))
        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{ins['level']}:</b> {ins['title']} — {ins['so_what']}",
                styles["Body"],
            ))
            story.append(Spacer(1, 8))

        # -------------------------
        # RECOMMENDATIONS
        # -------------------------
        story.append(Paragraph("Recommendations", styles["Section"]))
        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{rec['priority']}:</b> {rec['action']} ({rec['timeline']})",
                styles["Body"],
            ))
            story.append(Spacer(1, 6))

        # -------------------------
        # BUILD PDF
        # -------------------------
        doc.build(story)

        # ✅ SAFE CLEANUP (AFTER BUILD)
        for p in temp_images:
            try:
                os.remove(p)
            except Exception:
                pass

        return output_path

    # -------------------------
    # CHART RENDERER
    # -------------------------
    def _render_chart(self, vis: Dict[str, Any]) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.figure(figsize=(6, 4))
        plt.plot(vis.get("data", []), color="#2563eb")
        plt.title(vis.get("title", ""))
        plt.tight_layout()
        plt.savefig(tmp.name, dpi=150)
        plt.close()
        return tmp.name
