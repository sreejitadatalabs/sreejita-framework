# sreejita/reporting/html_renderer.py

from pathlib import Path
import markdown


class HTMLReportRenderer:
    """
    Converts Markdown reports into self-contained HTML.
    No external dependencies.
    """

    def render(self, md_path: Path, output_dir: Path | None = None) -> Path:
        if not md_path.exists():
            raise FileNotFoundError(md_path)

        if output_dir is None:
            output_dir = md_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        html_path = output_dir / md_path.with_suffix(".html").name

        md_text = md_path.read_text(encoding="utf-8")
        html_body = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code"]
        )

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Sreejita Executive Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #1f2937;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
        }}
        blockquote {{
            background: #f9fafb;
            padding: 10px 20px;
            border-left: 4px solid #3b82f6;
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
