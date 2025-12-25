DEFAULT_CONFIG = {
    # -----------------------------
    # DOMAIN (AUTO BY DEFAULT)
    # -----------------------------
    "domain": {
        "name": "auto",
    },

    # -----------------------------
    # LLM NARRATIVE (v3.5)
    # -----------------------------
    "narrative": {
        "enabled": False,          # OFF by default
        "provider": "openai",      # openai | gemini
        "model": "gpt-4o-mini",    # production default
        "confidence_band": "MEDIUM",
    },

    # -----------------------------
    # REPORTING
    # -----------------------------
    "report": {
        "mode": "hybrid",          # hybrid | executive | dynamic
        "format": "md",            # Markdown = source of truth
    },

    # -----------------------------
    # OUTPUT
    # -----------------------------
    "output_dir": "runs",

    # -----------------------------
    # METADATA
    # -----------------------------
    "metadata": {
        "framework": "Sreejita",
        "version": "v3.5",
    },
}
