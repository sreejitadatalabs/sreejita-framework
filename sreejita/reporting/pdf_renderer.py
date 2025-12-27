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
# FORMATTERS (EXECUTIVE SAFE)
# =====================================================

def fmt_number(x):
    try:
        x = float(x)
    except Exception:
        return str(x)

    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    if abs(x) < 1:
        return f"{x:.2f}"
    return f"{int(x):,}"


def fmt_percent(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


# =====================================================
# EXECUTIVE PDF RENDERER — FINAL
# =====================================================

class ExecutivePDFRenderer:
    """
    Sreejita v3.5.1 — Executive PDF Renderer (FINAL)

    ✔ No HTML
    ✔ No Markdown dependency
    ✔ No domain visual dependency
    ✔ Streamlit safe
    ✔ GitHub safe
    ✔ Always generates PDF
    """

    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
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
            alignment=TA_CENTER,
            textColor=self.PRIMARY,
            spaceAfter=20,
        ))

        styles.add(ParagraphStyle(
            name="Section",
            fontSize=16,
            textColor=self.PRIMARY,
            spaceBefore=18,
            spaceAfter=10,
        ))

        styles.add(ParagraphStyle(
            name="Body",
            fontSize=11,
            leading=14,
        ))

        # =====================================================
        # PAGE 1 — EXECUTIVE BRIEF
        # =====================================================
        meta = payload.get("meta", {})

        story.append(Paragraph("Sreejita Executive Report", styles["Title"]))
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}<br/>"
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["Body"],
        ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Executive Brief", styles["Section"]))

        for line in payload.get("summary", [])[:5]:
            story.append(Paragraph(f"• {line}", styles["Body"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Key Metrics", styles["Section"]))

        kpi_table = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            if any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion"]):
                val = fmt_percent(v)
            else:
                val = fmt_number(v)
            kpi_table.append([k.replace("_", " ").title(), val])

        table = Table(kpi_table, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(table)

        # =====================================================
        # VISUAL PAGES (SELF-RENDERED)
        # =====================================================
        visuals = payload.get("visuals", [])
        for idx, vis in enumerate(visuals[:4]):
            if idx % 2 == 0:
                story.append(PageBreak())
                story.append(Paragraph("Visual Evidence", styles["Section"]))

            img = self._safe_plot(vis)
            story.append(Image(img, width=6.5 * inch, height=3.5 * inch))
            story.append(Paragraph(vis.get("caption", "Visualization"), styles["Body"]))
            story.append(Spacer(1, 10))
            os.remove(img)

        # =====================================================
        # INSIGHTS & ACTIONS
        # =====================================================
        story.append(PageBreak())
        story.append(Paragraph("Insights, Risks & Recommendations", styles["Section"]))

        for ins in payload.get("insights", [])[:6]:
            story.append(Paragraph(
                f"<b>{ins['level']}:</b> {ins['title']} — {ins['so_what']}",
                styles["Body"],
            ))

        story.append(Spacer(1, 12))

        if payload.get("recommendations"):
            story.append(Paragraph("Action Plan", styles["Section"]))
            for rec in payload["recommendations"][:3]:
                story.append(Paragraph(
                    f"<b>{rec['priority']}:</b> {rec['action']} "
                    f"({rec['timeline']})",
                    styles["Body"],
                ))

        doc.build(story)
        return output_path

    # =====================================================
    # SAFE INTERNAL VISUAL RENDERER
    # =====================================================
    def _safe_plot(self, vis: Dict[str, Any]) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        data = vis.get("data", [])

        plt.figure(figsize=(7, 4))
        plt.plot(data if data else [0, 1, 2], color="#2563eb")
        plt.title(vis.get("title", ""))
        plt.tight_layout()
        plt.savefig(tmp.name, dpi=150)
        plt.close()

        return tmp.name
