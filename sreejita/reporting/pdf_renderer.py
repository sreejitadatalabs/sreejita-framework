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
    return f"{x:,.0f}"


def format_percent(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


# -------------------------------------------------
# EXECUTIVE PDF RENDERER
# -------------------------------------------------

class ExecutivePDFRenderer:
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
            name="ExecTitle", fontSize=22, alignment=TA_CENTER,
            spaceAfter=20, textColor=self.PRIMARY
        ))
        styles.add(ParagraphStyle(
            name="ExecSection", fontSize=16,
            spaceBefore=20, spaceAfter=10, textColor=self.PRIMARY
        ))
        styles.add(ParagraphStyle(
            name="ExecBody", fontSize=11, leading=14
        ))

        # -------- PAGE 1 --------
        story.append(Paragraph("Sreejita Executive Report", styles["ExecTitle"]))
        meta = payload.get("meta", {})
        story.append(Paragraph(f"<b>Domain:</b> {meta.get('domain','Unknown')}", styles["ExecBody"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}", styles["ExecBody"]))
        story.append(Spacer(1, 14))

        story.append(Paragraph("Executive Brief", styles["ExecSection"]))
        for line in payload.get("summary", []):
            story.append(Paragraph(f"• {line}", styles["ExecBody"]))

        story.append(Spacer(1, 14))
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            val = format_percent(v) if "rate" in k.lower() else format_number(v)
            table_data.append([k.replace("_", " ").title(), val])

        table = Table(table_data, colWidths=[3.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
            ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story.append(table)

        # -------- VISUALS --------
        visuals = payload.get("visuals", [])
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))

            for vis in visuals[:4]:
                img = self._render_chart(vis)
                story.append(Image(img, width=5.5*inch, height=3.2*inch))
                story.append(Paragraph(vis.get("caption",""), styles["ExecBody"]))
                os.remove(img)

        # -------- INSIGHTS --------
        story.append(PageBreak())
        story.append(Paragraph("Key Insights", styles["ExecSection"]))
        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{ins.get('level')}:</b> {ins.get('title')} — {ins.get('so_what')}",
                styles["ExecBody"]
            ))

        story.append(Spacer(1, 10))
        story.append(Paragraph("Recommendations", styles["ExecSection"]))
        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{rec.get('priority')}:</b> {rec.get('action')} ({rec.get('timeline')})",
                styles["ExecBody"]
            ))

        doc.build(story)
        return output_path

    # -------- SAFE CHART --------
    def _render_chart(self, vis: Dict[str, Any]) -> str:
        if not vis.get("x") or not vis.get("y"):
            raise ValueError(f"Invalid visual payload: {vis}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.figure(figsize=(6,4))

        if vis.get("type") == "bar":
            plt.bar(vis["x"], vis["y"])
        elif vis.get("type") == "hist":
            plt.hist(vis["x"], bins=10)
        else:
            plt.plot(vis["x"], vis["y"], marker="o")

        plt.title(vis.get("title",""))
        plt.xlabel(vis.get("xlabel",""))
        plt.ylabel(vis.get("ylabel",""))
        plt.tight_layout()
        plt.savefig(tmp.name, dpi=150)
        plt.close()
        return tmp.name
