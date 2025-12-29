import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib import utils
from reportlab.lib.units import inch


# =====================================================
# PDF PAYLOAD NORMALIZER (MANDATORY)
# =====================================================

def normalize_pdf_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict): payload = {}
    payload.setdefault("meta", {})
    payload.setdefault("summary", ["Operational performance indicators require management attention."])
    payload.setdefault("kpis", {})
    payload.setdefault("visuals", [])
    payload.setdefault("insights", [])
    payload.setdefault("recommendations", [])
    return payload


# =====================================================
# KPI FORMATTERS
# =====================================================

def format_number(x):
    if x is None or x == "": return "-"
    try:
        if pd.isna(x): return "-"
    except: pass
    try: x = float(x)
    except (ValueError, TypeError): return str(x)

    if abs(x) >= 1_000_000: return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000: return f"{x / 1_000:.1f}K"
    if abs(x) < 1 and x != 0: return f"{x:.2f}"
    try: return f"{int(x):,}"
    except ValueError: return str(x)


def format_percent(x):
    try:
        val = float(x)
        if pd.isna(val): return "-"
        return f"{val * 100:.1f}%"
    except Exception: return str(x)


# =====================================================
# EXECUTIVE PDF RENDERER (FINAL)
# =====================================================

class ExecutivePDFRenderer:
    PRIMARY = HexColor("#1f2937")
    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        payload = normalize_pdf_payload(payload)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        styles.add(ParagraphStyle(name="ExecTitle", fontName="Helvetica-Bold", fontSize=24, alignment=TA_CENTER, spaceAfter=24, textColor=self.PRIMARY))
        styles.add(ParagraphStyle(name="ExecSection", fontName="Helvetica-Bold", fontSize=16, spaceBefore=20, spaceAfter=12, textColor=self.PRIMARY))
        styles.add(ParagraphStyle(name="ExecBody", fontName="Helvetica", fontSize=11, leading=15, spaceAfter=8))
        styles.add(ParagraphStyle(name="ExecCaption", fontName="Helvetica-Oblique", fontSize=9, alignment=TA_CENTER, textColor=HexColor("#6b7280"), spaceAfter=12))

        # PAGE 1
        story.append(Paragraph("Sreejita Executive Report", styles["ExecTitle"]))
        story.append(Paragraph(f"<b>Domain:</b> {payload['meta'].get('domain', 'Unknown')}", styles["ExecBody"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}", styles["ExecBody"]))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Executive Brief", styles["ExecSection"]))
        for line in payload["summary"]:
            clean_line = line.lstrip("- ").lstrip("• ")
            story.append(Paragraph(f"• {clean_line}", styles["ExecBody"]))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        
        # 🛡️ HARD FILTER: Filter KPIs to remove objects/dicts
        safe_kpis = {}
        for k, v in payload["kpis"].items():
            if isinstance(v, (str, int, float, type(None))):
                safe_kpis[k] = v

        for k, v in list(safe_kpis.items())[:12]:
            is_rate = any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion", "yield"])
            val_fmt = format_percent(v) if is_rate else format_number(v)
            table_data.append([k.replace("_", " ").title(), val_fmt])

        if table_data:
            table = Table(table_data, colWidths=[4 * inch, 2 * inch])
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, self.BORDER), ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("PADDING", (0, 0), (-1, -1), 8), ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
            story.append(table)

        # PAGE 2-3 VISUALS
        visuals = payload["visuals"]
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))
            for vis in visuals[:4]:
                img_path = Path(vis.get("path", ""))
                if img_path.exists():
                    try:
                        img_reader = utils.ImageReader(str(img_path))
                        iw, ih = img_reader.getSize()
                        aspect = ih / float(iw)
                        display_width = 6 * inch
                        display_height = display_width * aspect
                        if display_height > 4 * inch:
                            display_height = 4 * inch
                            display_width = display_height / aspect
                        story.append(Image(str(img_path), width=display_width, height=display_height))
                        story.append(Paragraph(vis.get("caption", ""), styles["ExecCaption"]))
                        story.append(Spacer(1, 12))
                    except: pass

        # PAGE 4 INSIGHTS
        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))
        if not payload["insights"]: story.append(Paragraph("No critical risks detected.", styles["ExecBody"]))
        for ins in payload["insights"]:
            color_hex = "#dc2626" if ins['level'] == "CRITICAL" else "#ea580c" if ins['level'] == "RISK" else "#1f2937"
            story.append(Paragraph(f"<font color='{color_hex}'><b>{ins['level']}:</b></font> <b>{ins['title']}</b>", styles["ExecBody"]))
            story.append(Paragraph(f"{ins['so_what']}", styles["ExecBody"]))
            story.append(Spacer(1, 10))

        # PAGE 5 RECOMMENDATIONS
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", styles["ExecSection"]))
        if not payload["recommendations"]: story.append(Paragraph("Continue monitoring operations.", styles["ExecBody"]))
        for rec in payload["recommendations"]:
            story.append(Paragraph(f"<b>{rec.get('priority','HIGH')}:</b> {rec.get('action','Action required')}", styles["ExecBody"]))
            if rec.get("timeline"): story.append(Paragraph(f"<i>Timeline: {rec['timeline']}</i>", styles["ExecBody"]))
            if rec.get("owner"): story.append(Paragraph(f"<i>Owner: {rec['owner']}</i>", styles["ExecBody"]))
            if rec.get("expected_outcome") or rec.get("success_kpi"): 
                story.append(Paragraph(f"<i>Goal: {rec.get('expected_outcome') or rec.get('success_kpi')}</i>", styles["ExecBody"]))
            story.append(Spacer(1, 10))

        doc.build(story)
        if not output_path.exists(): raise RuntimeError("PDF build completed but file not found")
        return output_path
