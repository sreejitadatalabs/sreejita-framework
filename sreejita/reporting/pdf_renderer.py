import requests
from pathlib import Path
import logging

logger = logging.getLogger("sreejita.pdf")

BROWSERLESS_ENDPOINT = "https://chrome.browserless.io/pdf"
API_KEY = "<YOUR_API_KEY>"  # put in secrets later


class PDFRenderer:
    """
    Streamlit-safe PDF Renderer using Browserless (Chromium)
    """

    def render(self, html_path: Path, output_dir: Path) -> Path | None:
        html_path = Path(html_path)
        if not html_path.exists():
            logger.error("HTML not found: %s", html_path)
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / html_path.with_suffix(".pdf").name

        # Browserless needs a URL → so we serve file via file://
        html_url = html_path.resolve().as_uri()

        payload = {
            "url": html_url,
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
            resp = requests.post(
                f"{BROWSERLESS_ENDPOINT}?token={API_KEY}",
                json=payload,
                timeout=60,
            )

            resp.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(resp.content)

            return pdf_path

        except Exception as e:
            logger.warning("PDF generation failed: %s", e)
            return None
