# -------------------------------------------------
# Streamlit App — Sreejita Framework (Hybrid Demo)
# -------------------------------------------------

import sys
from pathlib import Path
import os
import uuid
import streamlit as st

# -------------------------------------------------
# FIX: Ensure project root is in PYTHONPATH
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
st.markdown("**Version:** UI v1.9 / Engine v2.4")
st.info("Demo build — v1 report stability with v2 intelligence (shadow mode).")
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
# 2️⃣ Configuration
# -------------------------------------------------
st.subheader("2️⃣ Configuration")

domain = st.selectbox(
    "Select domain",
    options=["Auto", "Retail"],
    help="Auto domain detection powered by v2.4 engine"
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
                # Temp directory
                temp_dir = Path("ui/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded file
                file_id = uuid.uuid4().hex[:8]
                input_path = temp_dir / f"{file_id}_{uploaded_file.name}"

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run backend
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

    # -----------------------------
    # v2 Intelligence Preview
    # -----------------------------
    with st.expander("🧠 Decision Intelligence (v2.4 preview)"):
        st.json({
            "selected_domain": result.get("domain"),
            "confidence": result.get("domain_confidence"),
            "rules_applied": result.get("decision_rules"),
            "fingerprint": result.get("decision_fingerprint"),
        })

    # -----------------------------
    # Run metadata
    # -----------------------------
    with st.expander("Run details"):
        st.json({
            "generated_at": result.get("generated_at"),
            "version": result.get("version")
        })
