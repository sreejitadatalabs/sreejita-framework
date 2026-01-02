import sys
from pathlib import Path
import uuid
import streamlit as st

# -------------------------------------------------
# PATH FIX (REQUIRED FOR STREAMLIT)
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
st.caption("v3.5.1 — Markdown Source · Executive PDF (Stable)")

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
st.caption("AI narratives are optional and never replace deterministic intelligence.")

provider = st.selectbox(
    "AI Provider",
    options=["gemini", "openai"],
)

export_pdf = st.checkbox(
    "📄 Export Executive PDF",
    value=True,
)

# -------------------------------------------------
# RUN
# -------------------------------------------------
if st.button("🚀 Run Analysis"):
    if not uploaded_file:
        st.error("Please upload a file first.")
        st.stop()

    try:
        with st.spinner("Running analysis…"):
            temp_dir = Path("ui/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_path = temp_dir / f"{uuid.uuid4().hex}_{uploaded_file.name}"
            temp_path.write_bytes(uploaded_file.getbuffer())

            result = run_analysis_from_ui(
                input_path=str(temp_path),
                narrative_enabled=enable_narrative,
                narrative_provider=provider,
                generate_pdf=export_pdf,
            )

    except Exception as e:
        st.error("❌ Analysis failed")
        st.exception(e)
        st.stop()

    # -------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------
    st.success("✅ Analysis complete")

    # ---- MARKDOWN ----
    if result.get("markdown") and Path(result["markdown"]).exists():
        with open(result["markdown"], "rb") as f:
            st.download_button(
                "📝 Download Markdown Report",
                f,
                file_name=Path(result["markdown"]).name,
                mime="text/markdown",
            )
    else:
        st.warning("⚠️ Markdown report not generated")

    # ---- PDF ----
    if export_pdf:
        if result.get("pdf") and Path(result["pdf"]).exists():
            with open(result["pdf"], "rb") as f:
                st.download_button(
                    "📄 Download Executive PDF",
                    f,
                    file_name=Path(result["pdf"]).name,
                    mime="application/pdf",
                )
        else:
            st.warning("⚠️ PDF was requested but not generated")

    st.caption(f"📁 Run folder: `{result.get('run_dir')}`")
