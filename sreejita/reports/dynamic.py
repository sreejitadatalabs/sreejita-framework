import os
import pandas as pd
from datetime import datetime, UTC

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
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
from sreejita.utils.logger import get_logger

log = get_logger(__name__)

def run_dynamic(input_path: str, output_path: str, config: dict):
    log.info("Running DYNAMIC report")

    df_raw = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_excel(input_path)

    result = clean_dataframe(df_raw, [config["dataset"].get("date")])
    df = result["df"]

        # Domain routing
    df = apply_domain(df, config["domain"]["name"])
    
    # Schema detection & column configuration
    schema = detect_schema(df)
    numeric_cols = config["analysis"].get("numeric") or schema["numeric"]
    categorical_cols = config["analysis"].get("categorical") or schema["categorical"]

    img_dir = "dynamic_images"
    os.makedirs(img_dir, exist_ok=True)

    images = []
    sales = config["dataset"].get("sales")

    if sales and config["dataset"].get("date"):
        trend = os.path.join(img_dir, "trend.png")
        plot_monthly(df, config["dataset"]["date"], sales, trend)
        images.append(trend)

    for cat in config["analysis"].get("categorical", [])[:2]:
        img = os.path.join(img_dir, f"{cat}.png")
        bar(df, cat, img)
        images.append(img)

    kpis = compute_kpis(df, sales, config["dataset"].get("profit"))
    insights = correlation_insights(df, sales)
    recs = generate_recommendations(df)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], alignment=1)
    h2 = styles["Heading2"]

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)

    story = []
    story.append(Paragraph("Dynamic Data Insight Report", h1))
    story.append(Spacer(1, 12))

    for k, v in kpis.items():
        story.append(Paragraph(f"<b>{k}</b>: {v}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Key Insights", h2))
    for i in insights:
        story.append(Paragraph(f"• {i}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Recommendations", h2))
    for r in recs:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    for img in images:
        story.append(Spacer(1, 12))
        story.append(Image(img, width=16*cm, height=5*cm))

    doc.build(story)
    log.info("Dynamic report generated → %s", output_path)
