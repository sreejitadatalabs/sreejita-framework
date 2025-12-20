import subprocess
from pathlib import Path
from typing import Optional


class PandocPDFRenderer:
    """
    Converts Markdown reports into PDF using Pandoc.
    This is a pure rendering layer (v3.x compatible).
    """

    def __init__(self, pandoc_path: str = "pandoc"):
        self.pandoc_path = pandoc_path

    def render(
        self,
        md_path: Path,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Convert a Markdown file to PDF.

        Parameters
        ----------
        md_path : Path
            Path to the generated Markdown report
        output_dir : Path | None
            Directory to place the PDF (defaults to MD parent)

        Returns
        -------
        Path
            Generated PDF file path
        """

        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        if output_dir is None:
            output_dir = md_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_dir / md_path.with_suffix(".pdf").name

        cmd = [
            self.pandoc_path,
            str(md_path),
            "-o",
            str(pdf_path),
            "--pdf-engine=xelatex",
            "--metadata",
            "title=Sreejita Executive Report",
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Pandoc PDF generation failed:\n{e.stderr.decode()}"
            )

        return pdf_path
