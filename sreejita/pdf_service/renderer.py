# renderer.py
from pathlib import Path
import tempfile
import uuid
from playwright.async_api import async_playwright


async def render_pdf_from_url(html_url: str) -> Path:
    """
    Renders PDF from a public HTML URL.
    """
    temp_dir = Path(tempfile.gettempdir())
    pdf_path = temp_dir / f"sreejita_{uuid.uuid4().hex}.pdf"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        try:
            page = await browser.new_page(
                viewport={"width": 1280, "height": 900}
            )

            await page.goto(
                html_url,
                wait_until="networkidle",
                timeout=60_000,
            )

            await page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={
                    "top": "20mm",
                    "bottom": "20mm",
                    "left": "15mm",
                    "right": "15mm",
                },
            )

        finally:
            await browser.close()

    return pdf_path
