import requests
from pathlib import Path
import logging
import os

logger = logging.getLogger("sreejita.pdf")

BROWSERLESS_ENDPOINT = "https://chrome.browserless.io/pdf"


class PDFRenderer:
    """
    v3.6 — External Chromium PDF Renderer (Browserless)

    - Streamlit Cloud compatible
    - GitHub Web compatible
    - No local Chromium / Playwright
    """

    def render(
        self,
        html_path: Path,
        output_dir: Path,
    ) -> Path | None:

        html_path = Path(html_path)
        if not html_path.exists():
            logger.error("HTML file not found: %s", html_path)
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_dir / html_path.with_suffix(".pdf").name

        # 🔐 Read API key safely
        api_key = os.environ.get("BROWSERLESS_API_KEY")
        if not api_key:
            logger.error("BROWSERLESS_API_KEY not set")
            return None

        # ✅ Read full HTML content
        html_content = html_path.read_text(encoding="utf-8")

        payload = {
            "html": html_content,
            "options": {
                "printBackground": True,
                "format": "A4",
                "margin": {
                    "top": "20mm",
                    "bottom": "20mm",
                    "left": "15mm",
                    "right": "15mm",
                },
            },
        }

        try:
            response = requests.post(
                f"{BROWSERLESS_ENDPOINT}?token={api_key}",
                json=payload,
                timeout=90,
            )

            response.raise_for_status()

            pdf_path.write_bytes(response.content)

            logger.info("PDF generated successfully: %s", pdf_path)
            return pdf_path

        except Exception as e:
            logger.warning("PDF generation failed: %s", e)
            return None
