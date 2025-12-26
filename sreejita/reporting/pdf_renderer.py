from pathlib import Path
from typing import Optional
import asyncio
import logging

logger = logging.getLogger("sreejita.pdf")


class PDFRenderer:
    """
    v3.6 Chromium-based PDF Renderer

    - HTML → PDF using Playwright (Chromium)
    - Fail-safe: never blocks report generation
    - SaaS-ready
    """

    def render(
        self,
        html_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Render PDF from an existing HTML file.

        Returns:
            Path to PDF if successful, else None
        """

        html_path = Path(html_path)
        if not html_path.exists():
            logger.error("HTML file not found: %s", html_path)
            return None

        output_dir = output_dir or html_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_dir / html_path.with_suffix(".pdf").name

        try:
            asyncio.run(self._render_async(html_path, pdf_path))
            if pdf_path.exists():
                logger.info("PDF generated: %s", pdf_path)
                return pdf_path
        except RuntimeError:
            # event loop already running (Streamlit)
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(
                    self._render_async(html_path, pdf_path)
                )
                if pdf_path.exists():
                    return pdf_path
            except Exception as e:
                logger.warning("PDF generation failed: %s", e)
        except Exception as e:
            logger.warning("PDF generation failed: %s", e)

        return None

    async def _render_async(self, html_path: Path, pdf_path: Path):
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            page = await browser.new_page()

            # Important: file:// URL
            await page.goto(
                html_path.resolve().as_uri(),
                wait_until="networkidle",
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

            await browser.close()
