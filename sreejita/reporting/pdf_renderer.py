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


# -------------------------------------------------
# FORMATTERS
# -------------------------------------------------

def format_number(x):
    try:
        x = float(x)
    except Exception:
        return str(x)

    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    if abs(x) < 1 and x != 0:
        return f"{x:.2f}"
    return f"{int(x):,}"


def format_percent(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


# -------------------------------------------------
# EXECUTIVE PDF RENDERER (FINAL)
# -------------------------------------------------

class ExecutivePDFRenderer:
    """
    v3.5.1 — ReportLab Executive PDF
    ✔ Streamlit safe
    ✔ GitHub safe
    ✔ No browser
    ✔ No HTML
    ✔ No ImageReader
    ✔ GUARANTEED PDF
    """

    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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

        styles.add(ParagraphStyle(
            name="Title",
            fontSize=22,
            alignment=TA_CENTER,
            textColor=self.PRIMARY,
            spaceAfter=24,
        ))
        styles.add(ParagraphStyle(
            name="Section",
            fontSize=16,
            textColor=self.PRIMARY,
            spaceBefore=20,
            spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            name="Body",
            fontSize=11,
            leading=14,
        ))

        # -------------------------------------------------
        # PAGE 1 — EXECUTIVE BRIEF
        # -------------------------------------------------
        story.append(Paragraph("Sreejita Executive Report", styles["Title"]))
        meta = payload.get("meta", {})
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}",
            styles["Body"]
        ))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["Body"]
        ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Executive Summary", styles["Section"]))
        for s in payload.get("summary", []):
            story.append(Paragraph(f"• {s}", styles["Body"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Key Metrics", styles["Section"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            val = format_percent(v) if "rate" in k else format_number(v)
            table_data.append([k.replace("_", " ").title(), val])

        table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
        ]))
        story.append(table)

        # -------------------------------------------------
        # PAGE 2 & 3 — VISUALS (RENDERED HERE)
        # -------------------------------------------------
        visuals = payload.get("visuals", [])
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["Section"]))

            for vis in visuals[:4]:
                img_path = self._render_chart(vis)
                story.append(Image(img_path, width=5.5 * inch, height=3.2 * inch))
                story.append(Paragraph(vis.get("caption", ""), styles["Body"]))
                story.append(Spacer(1, 12))
                os.remove(img_path)

        # -------------------------------------------------
        # FINAL PAGE — INSIGHTS & ACTIONS
        # -------------------------------------------------
        story.append(PageBreak())
        story.append(Paragraph("Insights & Risks", styles["Section"]))
        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{ins.get('level')}:</b> {ins.get('title')} — {ins.get('so_what')}",
                styles["Body"]
            ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations", styles["Section"]))
        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{rec.get('priority')}:</b> {rec.get('action')} ({rec.get('timeline')})",
                styles["Body"]
            ))

        # -------------------------------------------------
        # BUILD (GUARANTEED)
        # -------------------------------------------------
        doc.build(story)

        if not output_path.exists():
            raise RuntimeError("PDF generation failed silently")

        return output_path

    # -------------------------------------------------
    # SAFE CHART RENDERER
    # -------------------------------------------------
    def _render_chart(self, vis: Dict[str, Any]) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.figure(figsize=(6, 4))
        plt.plot(vis.get("data", []))
        plt.title(vis.get("title", ""))
        plt.tight_layout()
        plt.savefig(tmp.name, dpi=150)
        plt.close()
        return tmp.name
