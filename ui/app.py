import streamlit as st
import os
import uuid
from datetime import datetime

# IMPORTANT: only import backend adapter
from backend import run_analysis_from_ui


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Sreejita Framework",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("Sreejita Framework")
st.caption("Automated Data Analysis & Reporting")
st.markdown("**Version:** v1.9 (Demo)")
st.divider()


# -----------------------------
# Upload Section
# -----------------------------
st.subheader("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size / 1024:.1f} KB")


# -----------------------------
# Configuration Section (Light)
# -----------------------------
st.subheader("2️⃣ Configuration")

domain = st.selectbox(
    "Select domain",
    options=["Auto", "Retail"],
    help="Domain intelligence will activate in v2.0"
)

report_type = st.selectbox(
    "Report type",
    options=["Hybrid PDF"],
    index=0
)


# -----------------------------
# Run Analysis
# -----------------------------
st.subheader("3️⃣ Run Analysis")

run_clicked = st.button("🚀 Run Analysis", type="primary")

result = None

if run_clicked:
    if not uploaded_file:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Running analysis..."):
            # Create temp directory
            os.makedirs("ui/temp", exist_ok=True)

            # Save uploaded file
            file_id = uuid.uuid4().hex[:8]
            input_path = os.path.join(
                "ui/temp",
                f"{file_id}_{uploaded_file.name}"
            )

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                result = run_analysis_from_ui(
                    input_path=input_path,
                    domain=domain
                )
            except Exception as e:
                st.error("Analysis failed.")
                st.exception(e)


# -----------------------------
# Output Section
# -----------------------------
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

    # Metadata (nice for demo)
    with st.expander("Run details"):
        st.json({
            "rows": result.get("rows"),
            "columns": result.get("columns"),
            "domain": domain,
            "generated_at": result.get("generated_at"),
            "version": "1.9.0"
        })
