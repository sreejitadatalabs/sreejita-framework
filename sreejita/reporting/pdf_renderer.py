# sreejita/reporting/pdf_renderer.py

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

        # Custom Styles
        styles.add(ParagraphStyle(name="ExecTitle", fontName="Helvetica-Bold", fontSize=24, alignment=TA_CENTER, spaceAfter=24, textColor=self.PRIMARY))
        styles.add(ParagraphStyle(name="ExecSection", fontName="Helvetica-Bold", fontSize=16, spaceBefore=20, spaceAfter=12, textColor=self.PRIMARY))
        styles.add(ParagraphStyle(name="ExecBody", fontName="Helvetica", fontSize=11, leading=15, spaceAfter=8))
        styles.add(ParagraphStyle(name="ExecCaption", fontName="Helvetica-Oblique", fontSize=9, alignment=TA_CENTER, textColor=HexColor("#6b7280"), spaceAfter=12))
        
        # 🔥 NEW STYLE: Contextual interpretation text
        styles.add(ParagraphStyle(name="ScoreContext", fontName="Helvetica-Oblique", fontSize=10, textColor=HexColor("#4b5563"), alignment=TA_CENTER, spaceBefore=6))

        # --- PAGE 1: HEADER & SCORECARD ---
        story.append(Paragraph("Sreejita Executive Report", styles["ExecTitle"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}", styles["ExecBody"]))
        story.append(Spacer(1, 12))

        # 🔥 BOARD VIEW SCORECARD
        kpis = payload.get("kpis", {})
        score = kpis.get("board_confidence_score", "N/A")
        breakdown = kpis.get("board_score_breakdown", {}) 
        maturity = kpis.get("maturity_level", "Unknown")
        trend = kpis.get("board_confidence_trend", "→")
        domain = payload['meta'].get('domain', 'Healthcare')
        
        # GAP 1 & 5 FIX: Fetch Interpretations
        interpretation = kpis.get("board_confidence_interpretation", "")
        trend_expl = kpis.get("trend_explanation", "")

        # Dynamic Color Logic
        score_color = "#10b981" # Green
        try:
            score_val = float(score)
            if score_val < 70: score_color = "#dc2626" # Red
            elif score_val < 85: score_color = "#f59e0b" # Orange
        except: pass

        # Scorecard Table Data
        score_data = [
            [f"Board Confidence Score: {score}/100", f"Maturity Level: {maturity}"],
            [f"Trend: {trend} (vs prior period)", f"Domain: {domain}"]
        ]
        
        score_table = Table(score_data, colWidths=[3.5*inch, 3.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), HexColor("#f8fafc")),
            ('TEXTCOLOR', (0,0), (0,0), HexColor(score_color)), # Score highlights
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 12),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOX', (0,0), (-1,-1), 1.5, HexColor("#e5e7eb")), # Thicker box
            ('PADDING', (0,0), (-1,-1), 14),
        ]))
        story.append(score_table)

        # 🔥 NEW: Render Score Interpretation (Gap 1)
        if interpretation:
            story.append(Paragraph(interpretation, styles["ScoreContext"]))
        
        # 🔥 NEW: Render Trend Explanation (Gap 5)
        if trend_expl:
             story.append(Paragraph(f"<i>({trend_expl})</i>", styles["ScoreContext"]))

        # GAP 4: MATURITY LEGEND
        legend_style = ParagraphStyle(name="Legend", fontName="Helvetica-Oblique", fontSize=8, textColor=HexColor("#6b7280"), alignment=TA_CENTER)
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Maturity Levels: Bronze (<60): Reactive/Risk-Driven | Silver (60-80): Controlled/Benchmark-Aware | Gold (>80): Predictive/Optimized",
            legend_style
        ))
        story.append(Spacer(1, 10))

        # 🔧 SCORE BREAKDOWN TABLE
        if breakdown and isinstance(breakdown, dict):
            story.append(Spacer(1, 8))
            
            # Header
            bd_data = [[Paragraph("<b>Score Drivers</b>", styles["ExecBody"]), Paragraph("<b>Impact</b>", styles["ExecBody"])]]
            
            # Rows
            for reason, points in breakdown.items():
                color = "green" if points > 0 else "red"
                sign = "+" if points > 0 else ""
                bd_data.append([
                    Paragraph(f"{reason}", styles["ExecBody"]),
                    Paragraph(f"<font color='{color}'><b>{sign}{points}</b></font>", styles["ExecBody"])
                ])
            
            t2 = Table(bd_data, colWidths=[5*inch, 2*inch])
            t2.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, HexColor("#e5e7eb")),
                ('BACKGROUND', (0,0), (-1,0), HexColor("#f3f4f6")),
                ('PADDING', (0,0), (-1,-1), 4),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(t2)

        story.append(Spacer(1, 20))

        # --- PAGE 1: EXECUTIVE BRIEF ---
        story.append(Paragraph("Executive Brief", styles["ExecSection"]))
        for line in payload["summary"]:
            clean_line = line.lstrip("- ").lstrip("• ")
            story.append(Paragraph(f"• {clean_line}", styles["ExecBody"]))

        story.append(Spacer(1, 16))
        
        # --- PAGE 1: KEY METRICS ---
        story.append(Paragraph("Key Metrics", styles["ExecSection"]))

        table_data = [["Metric", "Value"]]
        
        # Filter KPIs to remove objects/dicts (Keep only clean scalars)
        safe_kpis = {}
        for k, v in payload["kpis"].items():
            if isinstance(v, (str, int, float, type(None))):
                safe_kpis[k] = v

        # Limit to top 12 metrics
        for k, v in list(safe_kpis.items())[:12]:
            is_rate = any(x in k.lower() for x in ["rate", "ratio", "margin", "conversion", "yield"])
            val_fmt = format_percent(v) if is_rate else format_number(v)
            table_data.append([k.replace("_", " ").title(), val_fmt])

        if len(table_data) > 1:
            table = Table(table_data, colWidths=[4 * inch, 2 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER), 
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG), 
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), 
                ("PADDING", (0, 0), (-1, -1), 8), 
                ("ALIGN", (1, 0), (1, -1), "RIGHT")
            ]))
            story.append(table)

        # --- PAGE 2-3: VISUALS ---
        visuals = payload["visuals"]
        if visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["ExecSection"]))
            # Limit to 6 visuals for clean layout
            for vis in visuals[:6]:
                img_path = Path(vis.get("path", ""))
                if img_path.exists():
                    try:
                        img_reader = utils.ImageReader(str(img_path))
                        iw, ih = img_reader.getSize()
                        aspect = ih / float(iw)
                        display_width = 6 * inch
                        display_height = display_width * aspect
                        
                        # Cap large images
                        if display_height > 5 * inch:
                            display_height = 5 * inch
                            display_width = display_height / aspect
                        
                        story.append(Image(str(img_path), width=display_width, height=display_height))
                        story.append(Paragraph(vis.get("caption", ""), styles["ExecCaption"]))
                        story.append(Spacer(1, 16))
                    except: pass

        # --- PAGE 4: INSIGHTS ---
        story.append(PageBreak())
        story.append(Paragraph("Key Insights & Risks", styles["ExecSection"]))
        if not payload["insights"]: story.append(Paragraph("No critical risks detected.", styles["ExecBody"]))
        
        for ins in payload["insights"]:
            color_hex = "#dc2626" if ins['level'] == "CRITICAL" else "#ea580c" if ins['level'] == "RISK" else "#1f2937"
            story.append(Paragraph(f"<font color='{color_hex}'><b>{ins['level']}:</b></font> <b>{ins['title']}</b>", styles["ExecBody"]))
            story.append(Paragraph(f"{ins['so_what']}", styles["ExecBody"]))
            story.append(Spacer(1, 10))

        # --- PAGE 5: RECOMMENDATIONS ---
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", styles["ExecSection"]))
        if not payload["recommendations"]: story.append(Paragraph("Continue monitoring operations.", styles["ExecBody"]))
        
        for rec in payload["recommendations"]:
            # Format Action Item
            story.append(Paragraph(f"<b>{rec.get('priority','HIGH')}:</b> {rec.get('action','Action required')}", styles["ExecBody"]))
            
            # Details block (Timeline, Owner, Goal)
            details = []
            if rec.get("timeline"): details.append(f"Timeline: {rec['timeline']}")
            if rec.get("owner"): details.append(f"Owner: {rec['owner']}")
            if rec.get("expected_outcome") or rec.get("success_kpi"): 
                details.append(f"Goal: {rec.get('expected_outcome') or rec.get('success_kpi')}")
            
            if details:
                story.append(Paragraph(f"<i>{' | '.join(details)}</i>", styles["ExecBody"]))
            
            story.append(Spacer(1, 12))

        # BUILD PDF
        doc.build(story)
        if not output_path.exists(): raise RuntimeError("PDF build completed but file not found")
        return output_path
