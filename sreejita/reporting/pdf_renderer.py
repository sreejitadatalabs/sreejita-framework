from pathlib import Path
from typing import Optional
import asyncio
import logging

logger = logging.getLogger("sreejita.pdf")


class PDFRenderer:
    """
    v3.6 Chromium-based PDF Renderer

    - HTML → PDF using Playwright (Chromium)
    - Never blocks report generation
    - Safe for CLI, Streamlit, Docker, CI
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
            self._run_async(self._render_async(html_path, pdf_path))
        except Exception as e:
            logger.warning("PDF generation failed: %s", e)
            return None

        if pdf_path.exists():
            logger.info("PDF generated: %s", pdf_path)
            return pdf_path

        return None

    # -------------------------------------------------
    # ASYNC SAFETY LAYER
    # -------------------------------------------------
    def _run_async(self, coro):
        """
        Runs async code safely across:
        - CLI
        - Streamlit
        - Jupyter
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Streamlit / Jupyter case
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result()
        else:
            asyncio.run(coro)

    # -------------------------------------------------
    # CORE RENDERER
    # -------------------------------------------------
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

            try:
                page = await browser.new_page(
                    viewport={"width": 1280, "height": 900}
                )

                await page.goto(
                    html_path.resolve().as_uri(),
                    wait_until="networkidle",
                    timeout=60_000,  # 60s hard stop
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
