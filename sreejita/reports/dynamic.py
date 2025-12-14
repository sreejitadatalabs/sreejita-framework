import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.insights import correlation_insights
from sreejita.core.schema import detect_schema
from sreejita.domains.router import apply_domain
from sreejita.core.recommendations import generate_recommendations
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.categorical import bar
from typing import Optional

def run(input_path: str, config: dict, output_path: Optional[str] = None):
    if output_path is None:
        output_path = f"dynamic_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    df_raw = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_excel(input_path)
    df = clean_dataframe(df_raw, [config["dataset"].get("date")])["df"]

    df = apply_domain(df, config["domain"]["name"])
    schema = detect_schema(df)

    img_dir = Path("dynamic_images")
    img_dir.mkdir(exist_ok=True)

    images = []
    sales = config["dataset"].get("sales")
    date = config["dataset"].get("date")

    if sales and date:
        img = img_dir / "trend.png"
        plot_monthly(df, date, sales, img)
        images.append(img)

    for cat in schema["categorical"][:2]:
        img = img_dir / f"{cat}.png"
        bar(df, cat, img)
        images.append(img)

    kpis = compute_kpis(df, sales, config["dataset"].get("profit"))
    insights = correlation_insights(df, sales)
    recs = generate_recommendations(df)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], alignment=1)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2.5 * cm, bottomMargin=2 * cm)

    story = [Paragraph("Dynamic Data Insight Report", h1), Spacer(1, 12)]

    for k, v in kpis.items():
        story.append(Paragraph(f"<b>{k}</b>: {v}", styles["BodyText"]))

    for i in insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    for r in recs:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    for img in images:
        story.append(Image(str(img), width=16 * cm, height=5 * cm))

    doc.build(story)
