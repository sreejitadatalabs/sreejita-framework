from pathlib import Path
from typing import Optional
import shutil
import re

import markdown


class HTMLReportRenderer:
    """
    Markdown → HTML renderer (v3.4)

    - Client-ready styling
    - Visual evidence auto-resolution
    - HTML is canonical output (PDF later)
    """

    IMG_REGEX = re.compile(r'<img[^>]+src="([^"]+)"')

    def render(
        self,
        md_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Path:

        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        md_path = Path(md_path)
        output_dir = output_dir or md_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        html_path = output_dir / md_path.with_suffix(".html").name

        md_text = md_path.read_text(encoding="utf-8")

        html_body = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code"],
        )

        visuals_dir = md_path.parent / "visuals"

        for img_src in self.IMG_REGEX.findall(html_body):
            img_name = Path(img_src).name
            for src in [
                md_path.parent / img_name,
                visuals_dir / img_name,
            ]:
                if src.exists():
                    target = output_dir / img_name
                    if not target.exists():
                        shutil.copy(src, target)
                    break

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sreejita Executive Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    margin: 40px;
    color: #2c3e50;
}}
h1, h2, h3 {{
    color: #1f2d3d;
}}
table {{
    border-collapse: collapse;
    margin-top: 12px;
    width: 100%;
}}
table, th, td {{
    border: 1px solid #ccc;
    padding: 8px;
}}
th {{
    background: #f4f6f8;
}}
img {{
    max-width: 100%;
    margin: 14px 0;
    border: 1px solid #ddd;
}}
blockquote {{
    background: #f9fafb;
    padding: 12px;
    border-left: 4px solid #4b6cb7;
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
