from pathlib import Path
from typing import Optional

import markdown


class HTMLReportRenderer:
    """
    Converts Markdown reports into standalone HTML.

    v3.3 SAFE:
    - Python 3.9 compatible
    - No side effects
    - Rendering-only layer
    """

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

        # Read Markdown
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Convert to HTML
        html_body = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code"],
        )

        # Simple standalone HTML wrapper
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sreejita Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
}}
table {{
    border-collapse: collapse;
}}
table, th, td {{
    border: 1px solid #999;
    padding: 6px;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return html_path
