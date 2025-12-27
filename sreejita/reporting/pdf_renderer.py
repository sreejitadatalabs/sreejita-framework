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
# FORMATTERS
# =====================================================

def fmt_number(v):
    try:
        v = float(v)
    except Exception:
        return str(v)

    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.1f}K"
    if abs(v) < 1 and v != 0:
        return f"{v:.2f}"
    return f"{int(v):,}"


def fmt_percent(v):
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return str(v)


# =====================================================
# EXECUTIVE PDF RENDERER (HARD-STABLE)
# =====================================================

class ExecutivePDFRenderer:
    """
    v3.5.1 FINAL PDF ENGINE

    - No ReportLab style collisions
    - Payload-driven
    - Visual-safe
    - Guaranteed PDF or explicit error
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

        # ---------- SAFE CUSTOM STYLES ----------
        styles.add(ParagraphStyle(
            name="ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=24,
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
        # PAGE 1 — EXECUTIVE BRIEF
        # =====================================================

        story.append(Paragraph("Sreejita Executive Report", styles["ExecTitle"]))
        story.append(Spacer(1, 10))

        meta = payload.get("meta", {})
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}",
            styles["ExecBody"]
        ))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["ExecBody"]
        ))

        story.append(Spacer(1, 14))
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            if any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion"]):
                val = fmt_percent(v)
            else:
                val = fmt_number(v)
            table_data.append([k.replace("_", " ").title(), val])

        table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
            ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(table)

        # =====================================================
        # PAGE 2–3 — VISUAL EVIDENCE
        # =====================================================

        visuals = payload.get("visuals", [])
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))

            for i, vis in enumerate(visuals):
                img = self._render_chart(vis)
                story.append(Image(img, width=5.5 * inch, height=3.2 * inch))
                story.append(Paragraph(vis.get("caption", ""), styles["ExecBody"]))
                story.append(Spacer(1, 12))
                os.remove(img)

                if (i + 1) % 2 == 0:
                    story.append(PageBreak())

        # =====================================================
        # PAGE 4 — INSIGHTS + RECOMMENDATIONS
        # =====================================================

        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))

        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{ins.get('level','INFO')}:</b> "
                f"{ins.get('title','')} — {ins.get('so_what','')}",
                styles["ExecBody"]
            ))
            story.append(Spacer(1, 8))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations", styles["ExecSection"]))

        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{rec.get('priority','HIGH')}:</b> "
                f"{rec.get('action','')} ({rec.get('timeline','Immediate')})",
                styles["ExecBody"]
            ))
            story.append(Spacer(1, 6))

        # =====================================================
        # BUILD (GUARANTEED)
        # =====================================================

        doc.build(story)

        if not output_path.exists():
            raise RuntimeError("PDF build finished but file not found")

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
