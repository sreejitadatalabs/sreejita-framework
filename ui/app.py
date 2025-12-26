import sys
from pathlib import Path
import uuid
import streamlit as st

# -------------------------------------------------
# PATH FIX (REQUIRED)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.backend import run_analysis_from_ui

# -------------------------------------------------
# UI CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Sreejita Framework",
    page_icon="📊",
    layout="centered",
)

st.title("Sreejita Framework")
st.caption("v3.6 — Deterministic Analytics with Optional AI Narrative")

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV / Excel",
    ["csv", "xlsx"],
)

enable_narrative = st.checkbox(
    "🤖 Enable AI Narrative",
    value=False,
)

provider = st.selectbox(
    "AI Provider",
    options=["gemini", "openai"],
    help="Gemini for testing, OpenAI for production",
)

export_pdf = st.checkbox(
    "📄 Export PDF (Chromium)",
    value=False,
)

# -------------------------------------------------
# RUN
# -------------------------------------------------
if st.button("🚀 Run Analysis"):
    if not uploaded_file:
        st.error("Please upload a file first.")
        st.stop()

    with st.spinner("Running analysis…"):
        temp_dir = Path("ui/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = temp_dir / f"{uuid.uuid4().hex}_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        result = run_analysis_from_ui(
            input_path=str(temp_path),
            narrative_enabled=enable_narrative,
            narrative_provider=provider,
            generate_pdf=export_pdf,
        )

    # -------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------
    st.success("✅ Analysis complete")

    if result.get("html"):
        with open(result["html"], "rb") as f:
            st.download_button(
                "🌐 Download HTML Report",
                f,
                file_name=Path(result["html"]).name,
                mime="text/html",
            )

    if result.get("pdf"):
        with open(result["pdf"], "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name=Path(result["pdf"]).name,
                mime="application/pdf",
            )

    st.caption(f"📁 Run folder: `{result['run_dir']}`")
