from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

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
# EXECUTIVE PDF RENDERER — FINAL & STABLE
# =====================================================

class ExecutivePDFRenderer:
    """
    ✔ Streamlit-safe
    ✔ GitHub-safe
    ✔ No browser
    ✔ No matplotlib
    ✔ Handles dict + object insights
    ✔ Uses domain-generated visuals
    ✔ Deterministic PDF
    """

    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    # -------------------------------------------------
    # SAFE FIELD ACCESS (🔥 ROOT CAUSE FIX)
    # -------------------------------------------------
    def _get(self, obj, name, default=""):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    # -------------------------------------------------
    # RENDER
    # -------------------------------------------------
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

        # ---------------- STYLES ----------------
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
            spaceBefore=20,
            spaceAfter=12,
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
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}",
            styles["ExecBody"]
        ))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["ExecBody"]
        ))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Executive Brief", styles["ExecSection"]))

        for line in payload.get("summary", []):
            story.append(Paragraph(f"• {line}", styles["ExecBody"]))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload.get("kpis", {}).items():
            val = (
                format_percent(v)
                if any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion"])
                else format_number(v)
            )
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

            for vis in visuals[:4]:
                img_path = Path(self._get(vis, "path"))
                if img_path.exists():
                    story.append(Image(str(img_path), width=5.5 * inch, height=3.2 * inch))
                    story.append(Paragraph(
                        self._get(vis, "caption", ""),
                        styles["ExecBody"]
                    ))
                    story.append(Spacer(1, 12))

        # =====================================================
        # PAGE 4 — INSIGHTS, RECOMMENDATIONS, RISKS
        # =====================================================
        story.append(PageBreak())

        # ---- INSIGHTS ----
        story.append(Paragraph("Key Insights", styles["ExecSection"]))
        for ins in payload.get("insights", []):
            story.append(Paragraph(
                f"<b>{self._get(ins,'level','INFO')}:</b> "
                f"{self._get(ins,'title','')} — "
                f"{self._get(ins,'so_what','')}",
                styles["ExecBody"]
            ))

        # ---- RECOMMENDATIONS ----
        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations", styles["ExecSection"]))
        for rec in payload.get("recommendations", []):
            story.append(Paragraph(
                f"<b>{self._get(rec,'priority','HIGH')}:</b> "
                f"{self._get(rec,'action','')} "
                f"({self._get(rec,'timeline','Immediate')})",
                styles["ExecBody"]
            ))

        # ---------------- BUILD ----------------
        doc.build(story)

        if not output_path.exists():
            raise RuntimeError("PDF build completed but file not found")

        return output_path 
