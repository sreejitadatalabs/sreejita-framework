# -------------------------------------------------
# Streamlit App — Sreejita Framework
# v3.3 SAFE (GitHub Web Compatible)
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

# -------------------------------------------------
# Backend adapter ONLY (no rendering logic here)
# -------------------------------------------------
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
st.markdown("**Version:** UI v3.3 / Engine v3.3")
st.info("Hybrid Intelligence • Markdown Source • HTML Delivery")
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
    help="Auto domain detection powered by v3 engine",
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
                # Temp directory for uploads
                temp_dir = Path("ui/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                file_id = uuid.uuid4().hex[:8]
                input_path = temp_dir / f"{file_id}_{uploaded_file.name}"

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = run_analysis_from_ui(
                    input_path=str(input_path),
                    domain=domain,
                )

            except Exception as e:
                st.error("❌ Analysis failed")
                st.exception(e)

# -------------------------------------------------
# 4️⃣ Output (CLIENT-SAFE DELIVERY)
# -------------------------------------------------
if result:
    st.divider()
    st.subheader("4️⃣ Output")

    st.success("✅ Analysis completed")

    # -----------------------------
    # HTML REPORT (PRIMARY)
    # -----------------------------
    html_path = result.get("html_report_path")

    if html_path and Path(html_path).exists():
        with open(html_path, "rb") as f:
            st.download_button(
                label="🌐 Download Report (HTML)",
                data=f,
                file_name=Path(html_path).name,
                mime="text/html",
            )
    else:
        st.info("HTML report not available for this run.")

    # -----------------------------
    # MARKDOWN REPORT (FALLBACK)
    # -----------------------------
    md_path = result.get("md_report_path")

    if md_path and Path(md_path).exists():
        with open(md_path, "rb") as f:
            st.download_button(
                label="📄 Download Report (Markdown)",
                data=f,
                file_name=Path(md_path).name,
                mime="text/markdown",
            )

    # -----------------------------
    # Decision Intelligence
    # -----------------------------
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
