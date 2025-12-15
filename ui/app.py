# -------------------------------------------------
# Streamlit App — Sreejita Framework v1.9
# -------------------------------------------------

import sys
from pathlib import Path
import os
import uuid
import streamlit as st
from datetime import datetime

# -------------------------------------------------
# FIX: Ensure project root is in PYTHONPATH
# (REQUIRED for Streamlit Cloud)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# Import backend adapter ONLY
# -------------------------------------------------
from ui.backend import run_analysis_from_ui


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Sreejita Framework",
    page_icon="📊",
    layout="centered"
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("Sreejita Framework")
st.caption("Automated Data Analysis & Reporting")
st.markdown("**Version:** v1.9 (Demo)")
st.info("Demo version — optimized for small to medium datasets.")
st.divider()


# -------------------------------------------------
# 1️⃣ Upload Dataset
# -------------------------------------------------
st.subheader("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")


# -------------------------------------------------
# 2️⃣ Configuration (Light / Placeholder)
# -------------------------------------------------
st.subheader("2️⃣ Configuration")

domain = st.selectbox(
    "Select domain",
    options=["Auto", "Retail"],
    help="Domain intelligence activates in v2.0"
)

report_type = st.selectbox(
    "Report type",
    options=["Hybrid PDF"],
    index=0
)


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
                # Ensure temp directory exists
                temp_dir = Path("ui/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded file
                file_id = uuid.uuid4().hex[:8]
                input_path = temp_dir / f"{file_id}_{uploaded_file.name}"

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Call backend adapter
                result = run_analysis_from_ui(
                    input_path=str(input_path),
                    domain=domain
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

    st.success("✅ Report generated successfully")

    report_path = result.get("report_path")

    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name=os.path.basename(report_path),
                mime="application/pdf"
            )

    # Run metadata (demo-friendly)
    with st.expander("Run details"):
        st.json({
            "domain": result.get("domain"),
            "generated_at": result.get("generated_at"),
            "version": result.get("version")
        })
