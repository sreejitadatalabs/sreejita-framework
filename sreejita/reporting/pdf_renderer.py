from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import tempfile
import io

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
from reportlab.lib.utils import ImageReader


# =====================================================
# FORMATTERS
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
# EXECUTIVE PDF RENDERER (FINAL)
# =====================================================

class ExecutivePDFRenderer:
    """
    Sreejita v3.5.1 — FINAL PDF Renderer

    ✔ No filesystem race conditions
    ✔ Images embedded in-memory
    ✔ Streamlit & GitHub safe
    ✔ Deterministic PDF generation
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
            name="ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=20,
            textColor=self.PRIMARY,
        ))

        styles.add(ParagraphStyle(
            name="ExecSection",
            fontSize=16,
            spaceBefore=18,
            spaceAfter=10,
            textColor=self.PRIMARY,
        ))

        styles.add(ParagraphStyle(
            name="ExecBody",
            fontSize=11,
            leading=14,
        ))

        # =====================================================
        # PAGE 1 — EXECUTIVE BRIEF + KPIs
        # =====================================================
        story.append(Paragraph("Sreejita Executive Report", styles["ExecTitle"]))
        meta = payload.get("meta", {})
        story.append(Paragraph(f"<b>Domain:</b> {meta.get('domain','Unknown')}", styles["ExecBody"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}", styles["ExecBody"]))
        story.append(Spacer(1, 14))

        story.append(Paragraph("Executive Brief (1-minute read)", styles["ExecSection"]))
        for line in payload.get("summary", []):
            story.append(Paragraph(f"• {line}", styles["ExecBody"]))

        story.append(Spacer(1, 14))
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            value = format_percent(v) if "rate" in k else format_number(v)
            table_data.append([k.replace("_"," ").title(), value])

        table = Table(table_data, colWidths=[3.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
            ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story.append(table)

        # =====================================================
        # VISUAL PAGES (2 visuals per page)
        # =====================================================
        visuals = payload.get("visuals", [])
        for i in range(0, len(visuals), 2):
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))

            for vis in visuals[i:i+2]:
                img = self._render_chart_in_memory(vis)
                story.append(Image(img, width=5.5*inch, height=3.2*inch))
                story.append(Paragraph(vis.get("caption",""), styles["ExecBody"]))
                story.append(Spacer(1, 12))

        # =====================================================
        # FINAL PAGE — INSIGHTS + RECOMMENDATIONS
        # =====================================================
        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))
        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{ins.get('level','INFO')}:</b> {ins.get('title','')} — {ins.get('so_what','')}",
                styles["ExecBody"]
            ))

        story.append(Spacer(1, 14))
        story.append(Paragraph("Recommended Actions", styles["ExecSection"]))
        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{rec.get('priority','HIGH')}:</b> {rec.get('action','')} ({rec.get('timeline','Immediate')})",
                styles["ExecBody"]
            ))

        # =====================================================
        # BUILD (SAFE)
        # =====================================================
        doc.build(story)
        return output_path

    # =====================================================
    # IN-MEMORY CHART RENDERER (NO FILESYSTEM)
    # =====================================================
    def _render_chart_in_memory(self, vis: Dict[str, Any]) -> ImageReader:
        buffer = io.BytesIO()
        plt.figure(figsize=(6, 4))
        plt.plot(vis.get("data", []))
        plt.title(vis.get("title",""))
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=150)
        plt.close()
        buffer.seek(0)
        return ImageReader(buffer)
