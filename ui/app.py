# -------------------------------------------------
# Streamlit App — Sreejita Framework
# v3.5 SAFE
# -------------------------------------------------

import sys
from pathlib import Path
import uuid
import streamlit as st

# -------------------------------------------------
# Ensure project root is in PYTHONPATH
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.backend import run_analysis_from_ui

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Sreejita Framework",
    page_icon="📊",
    layout="centered",
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("Sreejita Framework")
st.caption("Automated Data Analysis & Reporting")
st.markdown("**Version:** UI v3.5 / Engine v3.5")
st.info("Hybrid Intelligence • Deterministic Core • Optional AI Narrative")
st.divider()

# -------------------------------------------------
# 1️⃣ Upload Dataset
# -------------------------------------------------
st.subheader("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"],
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")

# -------------------------------------------------
# 2️⃣ Configuration
# -------------------------------------------------
st.subheader("2️⃣ Configuration")

domain = st.selectbox(
    "Select domain",
    options=["Auto", "Retail", "Finance", "HR", "Healthcare", "Supply Chain"],
)

st.divider()

# -------------------------------------------------
# v3.5 Narrative Toggle
# -------------------------------------------------
st.subheader("Optional Enhancements")

enable_narrative = st.checkbox(
    "🤖 Enable AI-Assisted Narrative (v3.5)",
    value=False,
    help=(
        "Adds an AI-generated explanation of existing decisions. "
        "Does NOT change insights or recommendations."
    ),
)

st.divider()

# -------------------------------------------------
# 3️⃣ Run Analysis
# -------------------------------------------------
st.subheader("3️⃣ Run Analysis")

run_clicked = st.button("🚀 Run Analysis", type="primary")
result = None

if run_clicked:
    if not uploaded_file:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Running analysis..."):
            try:
                temp_dir = Path("ui/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                file_id = uuid.uuid4().hex[:8]
                input_path = temp_dir / f"{file_id}_{uploaded_file.name}"

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = run_analysis_from_ui(
                    input_path=str(input_path),
                    domain=domain,
                    narrative_enabled=enable_narrative,
                )

            except Exception as e:
                st.error("❌ Analysis failed")
                st.exception(e)

# -------------------------------------------------
# 4️⃣ Output
# -------------------------------------------------
if result:
    st.divider()
    st.subheader("4️⃣ Output")

    st.success("✅ Analysis completed")

    html_path = result.get("html_report_path")
    if html_path and Path(html_path).exists():
        with open(html_path, "rb") as f:
            st.download_button(
                "🌐 Download Report (HTML)",
                f,
                file_name=Path(html_path).name,
                mime="text/html",
            )
    else:
        st.info("HTML report not available.")

    md_path = result.get("md_report_path")
    if md_path and Path(md_path).exists():
        with open(md_path, "rb") as f:
            st.download_button(
                "📄 Download Report (Markdown)",
                f,
                file_name=Path(md_path).name,
                mime="text/markdown",
            )

    with st.expander("🧠 Decision Intelligence"):
        st.json({
            "selected_domain": result.get("domain"),
            "confidence": result.get("domain_confidence"),
            "rules_applied": result.get("decision_rules"),
            "fingerprint": result.get("decision_fingerprint"),
        })

    with st.expander("Run details"):
        st.json({
            "generated_at": result.get("generated_at"),
            "version": result.get("version"),
        })
