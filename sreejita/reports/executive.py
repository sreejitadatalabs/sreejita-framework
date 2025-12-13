import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.recommendations import generate_recommendations
from sreejita.core.schema import detect_schema
from sreejita.domains.router import apply_domain
from sreejita.utils.logger import get_logger

log = get_logger(__name__)

def run_executive(input_path: str, output_path: str, config: dict):
    log.info("Running EXECUTIVE report")

    df_raw = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_excel(input_path)
    df = clean_dataframe(df_raw, [config["dataset"].get("date")])["df"]

        # Domain routing
    df = apply_domain(df, config["domain"]["name"])
    
    # Schema detection & column configuration
    schema = detect_schema(df)
    numeric_cols = config["analysis"].get("numeric") or schema["numeric"]
    categorical_cols = config["analysis"].get("categorical") or schema["categorical"]

    kpis = compute_kpis(df,
                        config["dataset"].get("sales"),
                        config["dataset"].get("profit"))
    recs = generate_recommendations(df)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], alignment=1)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)

    story = []
    story.append(Paragraph("Executive Summary Report", h1))
    story.append(Spacer(1, 12))

    for k, v in kpis.items():
        story.append(Paragraph(f"<b>{k}</b>: {v}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Key Recommendations", styles["Heading2"]))
    for r in recs[:3]:
        story.append(Paragraph(f"• {r}", styles["BodyText"]))

    doc.build(story)
    log.info("Executive report generated → %s", output_path)
