DEFAULT_CONFIG = {
    "domain": {"name": "auto"},

    "narrative": {
        "enabled": False,
        "provider": "openai",   # default for production
        "model": "gpt-4o-mini",
        "confidence_band": "MEDIUM",
    },

    "report": {
        "mode": "hybrid",
        "format": "md",
    },

    "output_dir": "runs",

    "metadata": {
        "framework": "Sreejita",
        "version": "v3.5",
    },
}
