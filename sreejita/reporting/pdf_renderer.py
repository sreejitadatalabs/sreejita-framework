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
# KPI FORMATTERS (EXECUTIVE SAFE)
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
# EXECUTIVE PDF RENDERER (v3.5.1 — FINAL, STABLE)
# =====================================================

class ExecutivePDFRenderer:
    """
    Sreejita v3.5.1 — Executive PDF Renderer (ReportLab)

    ✔ Streamlit-safe
    ✔ GitHub Web-safe
    ✔ No browser
    ✔ No async
    ✔ No dependency on domain visuals
    ✔ Guaranteed PDF generation
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # DOCUMENT
        # -------------------------------------------------
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

        # -------------------------------------------------
        # CUSTOM STYLES (NO NAME COLLISIONS)
        # -------------------------------------------------
        styles.add(
            ParagraphStyle(
                name="ExecTitle",
                fontSize=22,
                spaceAfter=24,
                alignment=TA_CENTER,
                textColor=self.PRIMARY,
            )
        )

        styles.add(
            ParagraphStyle(
                name="ExecSection",
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=self.PRIMARY,
            )
        )

        styles.add(
            ParagraphStyle(
                name="ExecBody",
                fontSize=11,
                leading=14,
            )
        )

        # -------------------------------------------------
        # COVER
        # -------------------------------------------------
        story.append(Paragraph("Sreejita Executive Report", styles["ExecTitle"]))
        story.append(Spacer(1, 12))

        meta = payload.get("meta", {})
        story.append(
            Paragraph(
                f"<b>Domain:</b> {meta.get('domain', 'Unknown')}",
                styles["ExecBody"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
                styles["ExecBody"],
            )
        )

        story.append(PageBreak())

        # -------------------------------------------------
        # EXECUTIVE SUMMARY
        # -------------------------------------------------
        story.append(Paragraph("Executive Summary", styles["ExecSection"]))

        for line in payload.get("summary", []):
            story.append(Paragraph(f"• {line}", styles["ExecBody"]))
            story.append(Spacer(1, 6))

        # -------------------------------------------------
        # KPI SNAPSHOT
        # -------------------------------------------------
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]

        for k, v in payload.get("kpis", {}).items():
            if any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion"]):
                value = format_percent(v)
            else:
                value = format_number(v)

            table_data.append([k.replace("_", " ").title(), value])

        table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                    ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(table)

        # -------------------------------------------------
        # INSIGHTS
        # -------------------------------------------------
        story.append(PageBreak())
        story.append(Paragraph("Insights & Risks", styles["ExecSection"]))

        for ins in payload.get("insights", []):
            story.append(
                Paragraph(
                    f"<b>{ins.get('level', 'INFO')}:</b> "
                    f"{ins.get('title', '')} — {ins.get('so_what', '')}",
                    styles["ExecBody"],
                )
            )
            story.append(Spacer(1, 8))

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------
        story.append(Paragraph("Recommendations", styles["ExecSection"]))

        for rec in payload.get("recommendations", []):
            story.append(
                Paragraph(
                    f"<b>{rec.get('priority', 'HIGH')}:</b> "
                    f"{rec.get('action', '')} "
                    f"({rec.get('timeline', 'Immediate')})",
                    styles["ExecBody"],
                )
            )
            story.append(Spacer(1, 6))

        # -------------------------------------------------
        # BUILD PDF (GUARANTEED)
        # -------------------------------------------------
        doc.build(story)

        if not output_path.exists():
            raise RuntimeError("PDF build completed but file not found")

        return output_path
