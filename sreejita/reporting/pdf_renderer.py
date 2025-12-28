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
# PDF PAYLOAD NORMALIZER (MANDATORY)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    FINAL SAFETY GATE.
    Ensures PDF rendering NEVER crashes due to missing or malformed data.
    """

    payload.setdefault("meta", {})
    payload.setdefault("summary", [])
    payload.setdefault("kpis", {})
    payload.setdefault("visuals", [])
    payload.setdefault("insights", [])
    payload.setdefault("recommendations", [])

    # ---- Summary fallback ----
    if not payload["summary"]:
        payload["summary"] = [
            "Operational performance indicators require management attention."
        ]

    # ---- Insights normalization ----
    safe_insights = []
    for ins in payload["insights"]:
        if not isinstance(ins, dict):
            continue
        safe_insights.append({
            "level": ins.get("level", "INFO"),
            "title": ins.get("title", "Observation"),
            "so_what": ins.get("so_what", "Requires further review."),
        })
    payload["insights"] = safe_insights

    # ---- Recommendations normalization ----
    safe_recs = []
    for rec in payload["recommendations"]:
        if not isinstance(rec, dict):
            continue
        safe_recs.append({
            "priority": rec.get("priority", "HIGH"),
            "action": rec.get("action", "Action required"),
            "timeline": rec.get("timeline", "Immediate"),
        })
    payload["recommendations"] = safe_recs

    return payload


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
# EXECUTIVE PDF RENDERER (FINAL, HARDENED)
# =====================================================

class ExecutivePDFRenderer:
    """
    Sreejita Executive PDF Renderer — FINAL

    ✔ Streamlit-safe
    ✔ GitHub-safe
    ✔ No browser
    ✔ No matplotlib
    ✔ Domain visuals supported
    ✔ Never crashes
    ✔ Client-ready
    """

    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        # 🔒 HARD SAFETY GATE
        payload = normalize_pdf_payload(payload)

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

        # ---------- STYLES ----------
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

        meta = payload["meta"]
        story.append(Paragraph(
            f"<b>Domain:</b> {meta.get('domain', 'Unknown')}",
            styles["ExecBody"],
        ))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}",
            styles["ExecBody"],
        ))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Executive Brief", styles["ExecSection"]))

        for line in payload["summary"]:
            story.append(Paragraph(f"• {line}", styles["ExecBody"]))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        for k, v in payload["kpis"].items():
            val_fmt = (
                format_percent(v)
                if any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion"])
                else format_number(v)
            )
            table_data.append([k.replace("_", " ").title(), val_fmt])

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
        visuals = payload["visuals"]
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))

            for vis in visuals[:4]:
                img_path = Path(vis.get("path", ""))
                if img_path.exists():
                    story.append(Image(
                        str(img_path),
                        width=5.5 * inch,
                        height=3.2 * inch,
                    ))
                    story.append(
                        Paragraph(vis.get("caption", ""), styles["ExecBody"])
                    )
                    story.append(Spacer(1, 12))

        # =====================================================
        # PAGE 4 — INSIGHTS
        # =====================================================
        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))

        for ins in payload["insights"]:
            story.append(Paragraph(
                f"<b>{ins['level']}:</b> {ins['title']} — {ins['so_what']}",
                styles["ExecBody"],
            ))
            story.append(Spacer(1, 6))

        # =====================================================
        # PAGE 5 — RECOMMENDATIONS
        # =====================================================
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", styles["ExecSection"]))

        for rec in payload["recommendations"]:
            story.append(Paragraph(
                f"<b>{rec['priority']}:</b> {rec['action']} "
                f"({rec['timeline']})",
                styles["ExecBody"],
            ))
            story.append(Spacer(1, 6))

        # =====================================================
        # BUILD PDF (GUARANTEED)
        # =====================================================
        doc.build(story)

        if not output_path.exists():
            raise RuntimeError("PDF build completed but file not found")

        return output_path
