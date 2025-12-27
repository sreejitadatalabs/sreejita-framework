from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import tempfile

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

def fmt_number(x):
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


def fmt_percent(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)


# =====================================================
# EXECUTIVE PDF RENDERER (FINAL, STABLE)
# =====================================================

class ExecutivePDFRenderer:
    """
    Sreejita v3.5.1 — Executive PDF Renderer (ReportLab)

    ✔ Streamlit safe
    ✔ GitHub safe
    ✔ No async
    ✔ No HTML
    ✔ No external visuals dependency
    ✔ Deterministic
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
            spaceBefore=20,
            spaceAfter=12,
        ))

        styles.add(ParagraphStyle(
            name="Body",
            fontSize=11,
            leading=14,
        ))

        # =====================================================
        # PAGE 1 — EXECUTIVE BRIEF + KPIs
        # =====================================================
        meta = payload.get("meta", {})

        story.append(Paragraph("Sreejita Executive Report", styles["Title"]))
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}<br/>"
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["Body"]
        ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Executive Brief", styles["Section"]))

        for s in payload.get("summary", [])[:5]:
            story.append(Paragraph(f"• {s}", styles["Body"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Key Metrics", styles["Section"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            val = fmt_percent(v) if any(x in k.lower() for x in ["rate", "margin", "ratio"]) else fmt_number(v)
            table_data.append([k.replace("_", " ").title(), val])

        table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, self.BORDER),
            ("BACKGROUND", (0,0), (-1,0), self.HEADER_BG),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))

        story.append(table)

        # =====================================================
        # VISUAL PAGES (2 PER PAGE)
        # =====================================================
        visuals = payload.get("visuals", [])
        temp_images: List[str] = []

        for idx, vis in enumerate(visuals):
            if idx % 2 == 0:
                story.append(PageBreak())
                story.append(Paragraph("Visual Evidence", styles["Section"]))

            img_path = self._render_chart(vis)
            temp_images.append(img_path)

            story.append(Image(img_path, width=5.5 * inch, height=3.2 * inch))
            story.append(Paragraph(vis.get("caption", ""), styles["Body"]))
            story.append(Spacer(1, 12))

        # =====================================================
        # INSIGHTS & RECOMMENDATIONS
        # =====================================================
        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["Section"]))

        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{ins['level']}:</b> {ins['title']} — {ins['so_what']}",
                styles["Body"]
            ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations", styles["Section"]))

        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{rec['priority']}:</b> {rec['action']} ({rec['timeline']})",
                styles["Body"]
            ))

        # =====================================================
        # BUILD PDF (IMAGES MUST STILL EXIST HERE)
        # =====================================================
        doc.build(story)

        # Cleanup AFTER build
        for p in temp_images:
            try:
                Path(p).unlink()
            except Exception:
                pass

        return output_path

    # -------------------------
    # SAFE CHART RENDERER
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
