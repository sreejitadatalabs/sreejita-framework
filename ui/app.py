import sys
from pathlib import Path
import uuid
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.backend import run_analysis_from_ui

st.set_page_config(page_title="Sreejita Framework", page_icon="📊")
st.title("Sreejita Framework")
st.caption("v3.6 — HTML Primary + Optional PDF Export")

uploaded_file = st.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])

enable_narrative = st.checkbox("🤖 Enable AI Narrative", value=False)

provider = st.selectbox(
    "AI Provider",
    options=["gemini", "openai"],
)

generate_pdf = st.checkbox("📄 Generate PDF (v3.6)", value=False)

if st.button("🚀 Run Analysis"):
    if not uploaded_file:
        st.error("Upload a file first")
    else:
        temp = Path("ui/temp")
        temp.mkdir(parents=True, exist_ok=True)

        path = temp / f"{uuid.uuid4().hex}_{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Running analysis…"):
            result = run_analysis_from_ui(
                input_path=str(path),
                narrative_enabled=enable_narrative,
                narrative_provider=provider,
                generate_pdf=generate_pdf,
            )

        st.success("Report generated successfully")

        # ---- HTML DOWNLOAD ----
        if result.get("html_report_path"):
            with open(result["html_report_path"], "rb") as f:
                st.download_button(
                    "🌐 Download HTML Report",
                    f,
                    file_name=Path(result["html_report_path"]).name,
                    mime="text/html",
                )

        # ---- PDF DOWNLOAD (OPTIONAL) ----
        if result.get("pdf_report_path"):
            with open(result["pdf_report_path"], "rb") as f:
                st.download_button(
                    "📄 Download PDF Report",
                    f,
                    file_name=Path(result["pdf_report_path"]).name,
                    mime="application/pdf",
                )
