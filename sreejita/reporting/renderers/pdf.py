from pathlib import Path
import subprocess
import logging

log = logging.getLogger("sreejita.pdf-renderer")


def render_pdf(md_path: Path) -> Path:
    """
    Converts a Markdown report into PDF using Pandoc.

    Requirements:
    - pandoc installed
    - LaTeX or wkhtmltopdf backend available
    """

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown report not found: {md_path}")

    pdf_path = md_path.with_suffix(".pdf")

    try:
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-o",
                str(pdf_path),
                "--pdf-engine=xelatex",
            ],
            check=True,
        )
        log.info("PDF generated: %s", pdf_path.name)
        return pdf_path

    except subprocess.CalledProcessError as e:
        log.error("PDF generation failed: %s", e)
        raise RuntimeError("PDF rendering failed") from e
