DEFAULT_CONFIG = {
    # -----------------------------
    # DATASET METADATA (OPTIONAL)
    # -----------------------------
    "dataset": {
        "date": None,
        "target": None,
    },

    # -----------------------------
    # ANALYSIS HINTS (OPTIONAL)
    # -----------------------------
    "analysis": {
        "numeric": [],
        "categorical": [],
    },

    # -----------------------------
    # DOMAIN OVERRIDES (OPTIONAL)
    # -----------------------------
    "domain": {
        "name": "auto",  # auto-detect by default
    },
    
    # -----------------------------
    # LLM NARRATIVE
    # -----------------------------
    "narrative": {
    "enabled": False,      # OFF by default
    "provider": "openai",
    "model": "gpt-4o-mini",
    "confidence_band": "MEDIUM",
    },

    # -----------------------------
    # REPORTING (v3.3)
    # -----------------------------
    "report": {
        "mode": "hybrid",      # hybrid | executive | dynamic
        "format": "md",        # md is SOURCE OF TRUTH
    },

    # -----------------------------
    # OUTPUT CONTROL (CRITICAL)
    # -----------------------------
    "output_dir": "runs",

    # -----------------------------
    # DELIVERY (OPTIONAL)
    # -----------------------------
    # NOTE:
    # - HybridReport does NOT generate PDF
    # - CLI / Batch / GitHub Actions MAY
    "export_pdf": False,      # SAFE DEFAULT (no pandoc dependency)

    # -----------------------------
    # METADATA (OPTIONAL)
    # -----------------------------
    "metadata": {
        "framework": "Sreejita",
        "version": "v3.3",
    },
}
