from pathlib import Path
from typing import Optional
import shutil
import re

import markdown


class HTMLReportRenderer:
    """
    Converts Markdown reports into standalone HTML
    with visual evidence support.

    v3.3 SAFE:
    - Python 3.9 compatible
    - Copies images automatically
    - No side effects
    """

    IMAGE_PATTERN = re.compile(r'<img[^>]+src="([^"]+)"')

    def render(
        self,
        md_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Path:
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        if output_dir is None:
            output_dir = md_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        html_path = output_dir / md_path.with_suffix(".html").name

        # ----------------------------
        # Read Markdown
        # ----------------------------
        md_text = md_path.read_text(encoding="utf-8")

        # ----------------------------
        # Convert Markdown → HTML
        # ----------------------------
        html_body = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code"],
        )

        # ----------------------------
        # Resolve and copy images
        # ----------------------------
        md_dir = md_path.parent

        for img_src in self.IMAGE_PATTERN.findall(html_body):
            img_path = md_dir / img_src

            if img_path.exists():
                target_path = output_dir / img_path.name
                if not target_path.exists():
                    shutil.copy(img_path, target_path)

        # ----------------------------
        # Wrap in clean HTML template
        # ----------------------------
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sreejita Executive Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
}}
h1, h2, h3 {{
    color: #2c3e50;
}}
table {{
    border-collapse: collapse;
    margin-top: 10px;
}}
table, th, td {{
    border: 1px solid #999;
    padding: 6px;
}}
img {{
    max-width: 100%;
    margin: 10px 0;
    border: 1px solid #ddd;
}}
blockquote {{
    background: #f7f7f7;
    padding: 10px;
    border-left: 4px solid #999;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

        html_path.write_text(html, encoding="utf-8")
        return html_path
