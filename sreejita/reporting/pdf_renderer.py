# sreejita/reporting/pdf_renderer_reportlab.py

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


class ExecutivePDFRenderer:
    """
    Sreejita v3.5.1 — Executive PDF Renderer (ReportLab)

    ✔ Streamlit safe
    ✔ GitHub safe
    ✔ Client ready
    """

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
        styles.add(
            ParagraphStyle(
                name="TitleStyle",
                fontSize=22,
                spaceAfter=20,
                alignment=TA_CENTER,
            )
        )

        styles.add(
            ParagraphStyle(
                name="Section",
                fontSize=16,
                spaceBefore=20,
                spaceAfter=10,
                textColor=HexColor("#1f2937"),
            )
        )

        styles.add(
            ParagraphStyle(
                name="Body",
                fontSize=11,
                leading=14,
            )
        )

        # -------------------------
        # COVER PAGE
        # -------------------------
        story.append(Paragraph("Sreejita Executive Report", styles["TitleStyle"]))
        story.append(Spacer(1, 12))

        meta = payload.get("meta", {})
        story.append(Paragraph(f"<b>Domain:</b> {meta.get('domain', 'Unknown')}", styles["Body"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Body"]))
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

        kpis = payload.get("kpis", {})
        table_data = [["Metric", "Value"]]
        for k, v in kpis.items():
            table_data.append([k.replace("_", " ").title(), str(v)])

        table = Table(table_data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d1d5db")),
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f3f4f6")),
                ]
            )
        )

        story.append(table)

        # -------------------------
        # VISUAL EVIDENCE
        # -------------------------
        visuals = payload.get("visuals", [])
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["Section"]))

            for vis in visuals:
                img_path = self._create_chart(vis)
                story.append(Image(img_path, width=5.5 * inch, height=3.5 * inch))
                story.append(Paragraph(vis.get("caption", ""), styles["Body"]))
                story.append(Spacer(1, 12))
                os.remove(img_path)

        # -------------------------
        # INSIGHTS
        # -------------------------
        story.append(PageBreak())
        story.append(Paragraph("Insights & Risks", styles["Section"]))

        for ins in payload.get("insights", []):
            story.append(
                Paragraph(
                    f"<b>{ins['level']}:</b> {ins['title']} — {ins['so_what']}",
                    styles["Body"],
                )
            )
            story.append(Spacer(1, 8))

        # -------------------------
        # RECOMMENDATIONS
        # -------------------------
        story.append(Paragraph("Recommendations", styles["Section"]))
        for rec in payload.get("recommendations", []):
            story.append(
                Paragraph(
                    f"<b>{rec['priority']}:</b> {rec['action']} ({rec['timeline']})",
                    styles["Body"],
                )
            )
            story.append(Spacer(1, 6))

        # -------------------------
        # BUILD PDF
        # -------------------------
        doc.build(story)
        return output_path

    # -------------------------
    # CHART HELPER
    # -------------------------
    def _create_chart(self, vis: Dict[str, Any]) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.figure(figsize=(6, 4))
        plt.plot(vis.get("data", []))
        plt.title(vis.get("title", ""))
        plt.tight_layout()
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name
