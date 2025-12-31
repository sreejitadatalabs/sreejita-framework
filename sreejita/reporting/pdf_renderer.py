from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.units import inch
from reportlab.lib import utils


# =====================================================
# PAYLOAD NORMALIZER (STRICT & NESTED-SAFE)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens orchestrator → executive payload safely.
    NEVER crashes.
    """
    payload = payload if isinstance(payload, dict) else {}

    # Domain-wrapped payload safety
    if len(payload) == 1:
        payload = next(iter(payload.values()), payload)

    executive = payload.get("executive", {}) or {}

    return {
        "meta": payload.get("meta", {}),
        "executive_snapshot": (
            payload.get("executive_snapshot")
            or executive.get("snapshot")
            or {}
        ),
        "primary_kpis": (
            payload.get("primary_kpis")
            or executive.get("primary_kpis", [])
        ),
        "summary": payload.get("summary", []) or executive.get("summary", []),
        "visuals": payload.get("visuals", []),
        "insights": payload.get("insights", []),
        "recommendations": payload.get("recommendations", []),
    }


# =====================================================
# KPI FORMATTER (EXECUTIVE-SAFE)
# =====================================================

def format_kpi_value(key: str, value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, (int, float)):
        k = key.lower()

        if "rate" in k or "ratio" in k:
            return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"

        if "los" in k or "days" in k or "duration" in k:
            return f"{value:.1f} days"

        if "cost" in k or "billing" in k or "amount" in k:
            if value >= 1_000_000:
                return f"${value / 1_000_000:.1f}M"
            if value >= 1_000:
                return f"${value / 1_000:.0f}K"
            return f"${value:,.0f}"

        return f"{value:,.0f}"

    return str(value)


# =====================================================
# EXECUTIVE PDF RENDERER (FINAL — PRODUCTION)
# =====================================================

class ExecutivePDFRenderer:
    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        payload = normalize_pdf_payload(payload)

        # Sort visuals by importance
        payload["visuals"] = sorted(
            payload.get("visuals", []),
            key=lambda x: x.get("importance", 0),
            reverse=True,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        story: List[Any] = []

        # -------------------------------------------------
        # STYLES
        # -------------------------------------------------
        def add_style(name, **kwargs):
            if name not in styles:
                styles.add(ParagraphStyle(name=name, **kwargs))

        add_style(
            "ExecTitle",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=18,
            fontName="Helvetica-Bold",
            textColor=self.PRIMARY,
        )
        add_style(
            "ExecSection",
            fontSize=15,
            spaceBefore=18,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        )
        add_style(
            "ExecBody",
            fontSize=11,
            leading=15,
            spaceAfter=6,
        )
        add_style(
            "ExecCaption",
            fontSize=9,
            alignment=TA_CENTER,
            textColor=HexColor("#6b7280"),
            spaceAfter=12,
        )

        # =================================================
        # COVER PAGE
        # =================================================
        story.append(Paragraph("SREEJITA INTELLIGENCE FRAMEWORK™", styles["ExecTitle"]))
        story.append(Paragraph("Executive Healthcare Performance Report", styles["ExecSection"]))

        snapshot = payload.get("executive_snapshot", {})
        risk = snapshot.get("overall_risk", "-")

        story.append(Paragraph(
            f"Domain: Healthcare<br/>"
            f"Confidence Level: {risk}<br/>"
            f"Generated: {datetime.utcnow():%Y-%m-%d}",
            styles["ExecBody"],
        ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Prepared by: <b>Sreejita Data Labs</b>", styles["ExecBody"]))
        story.append(PageBreak())

        # =================================================
        # EXECUTIVE SNAPSHOT
        # =================================================
        if snapshot:
            story.append(Paragraph("Executive Decision Snapshot", styles["ExecSection"]))
            story.append(Paragraph(f"<b>Overall Risk:</b> {risk}", styles["ExecBody"]))

            for section, items, bullet in [
                ("Top Problems", snapshot.get("top_problems", []), "•"),
                ("Decisions Required", snapshot.get("decisions_required", []), "☐"),
            ]:
                if items:
                    story.append(Spacer(1, 8))
                    story.append(Paragraph(f"<b>{section}:</b>", styles["ExecBody"]))
                    for x in items[:3]:
                        story.append(Paragraph(f"{bullet} {x}", styles["ExecBody"]))

            story.append(PageBreak())

        # =================================================
        # PRIMARY KPIs (TOP 3–5)
        # =================================================
        primary = payload.get("primary_kpis", [])
        if primary:
            rows = [["Metric", "Value"]]
            for item in primary[:5]:
                rows.append([
                    item.get("name", "Metric"),
                    format_kpi_value(item.get("name", ""), item.get("value")),
                ])

            story.append(Paragraph("Key Performance Indicators", styles["ExecSection"]))
            table = Table(rows, colWidths=[4 * inch, 2 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(table)
            story.append(PageBreak())

        # =================================================
        # VISUALS (MAX 6)
        # =================================================
        if payload["visuals"]:
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))
            for vis in payload["visuals"][:6]:
                path = Path(vis.get("path", ""))
                if not path.exists():
                    continue
                try:
                    img = utils.ImageReader(str(path))
                    iw, ih = img.getSize()
                    w = 6 * inch
                    h = min((ih / iw) * w, 5 * inch)
                    story.append(Image(str(path), width=w, height=h))
                    story.append(Paragraph(vis.get("caption", ""), styles["ExecCaption"]))
                except Exception:
                    continue
            story.append(PageBreak())

        # =================================================
        # INSIGHTS (TOP 3–5)
        # =================================================
        if payload["insights"]:
            story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))
            for i in payload["insights"][:5]:
                story.append(Paragraph(
                    f"<b>{i.get('level','INFO')}:</b> {i.get('title','')}",
                    styles["ExecBody"],
                ))
                story.append(Paragraph(i.get("so_what", ""), styles["ExecBody"]))
                story.append(Spacer(1, 8))
            story.append(PageBreak())

        # =================================================
        # RECOMMENDATIONS (TOP 5)
        # =================================================
        if payload["recommendations"]:
            story.append(Paragraph("Recommendations", styles["ExecSection"]))
            for idx, r in enumerate(payload["recommendations"][:5], start=1):
                story.append(Paragraph(
                    f"{idx}. {r.get('action','Action required')}",
                    styles["ExecBody"],
                ))
                meta = " | ".join(filter(None, [
                    f"Owner: {r.get('owner')}",
                    f"Timeline: {r.get('timeline')}",
                    f"Outcome: {r.get('expected_outcome')}",
                ]))
                if meta:
                    story.append(Paragraph(meta, styles["ExecCaption"]))
                story.append(Spacer(1, 10))

        doc.build(story)
        return output_path
