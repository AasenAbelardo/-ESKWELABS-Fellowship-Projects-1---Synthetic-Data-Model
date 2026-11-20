"""Print README.md and assumptions.md as PDFs in the SD-147 palette.

    .venv/bin/python report/build_markdown_pdfs.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

DOCS = (
    ("README.md", "SD-147_README.pdf", "How to run it"),
    ("assumptions.md", "SD-147_Assumptions.pdf", "The specification"),
)

CSS = """
@page {
  size: A4;
  margin: 16mm 16mm 18mm;
  background: #F6F1E8;
  @top-left {
    content: "SD-147  ·  Urban air quality";
    font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif;
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #78716C;
  }
  @top-right {
    content: "Eskwelabs EIF";
    font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif;
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #0F6E6B;
  }
  @bottom-right {
    content: counter(page);
    font-family: "Iowan Old Style", Palatino, Georgia, serif;
    font-size: 11px;
    color: #78716C;
  }
}
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
html, body { margin: 0; background: #F6F1E8; color: #1C1917; }
body { font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif; font-size: 12px; line-height: 1.5; }
h1, h2, h3 { font-family: "Iowan Old Style", Palatino, Georgia, serif; font-weight: normal; letter-spacing: -0.02em; page-break-after: avoid; break-after: avoid; }
h1 { font-size: 34px; line-height: 1.05; margin: 2mm 0 5mm; }
h2 { font-size: 22px; line-height: 1.15; margin: 8mm 0 3mm; }
h3 { font-size: 16px; margin: 6mm 0 2mm; color: #0F6E6B; }
p { margin: 0 0 3.2mm; }
strong { font-weight: 600; }
a { color: #0F6E6B; text-decoration: none; }
code {
  font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  font-size: 0.9em;
  background: #EFE8DC;
  padding: 0.05em 0.32em;
  border-radius: 4px;
}
pre {
  background: #EFE8DC;
  border-radius: 8px;
  padding: 4mm 5mm;
  margin: 2mm 0 4mm;
  page-break-inside: avoid;
  break-inside: avoid;
  white-space: pre-wrap;
}
pre code { background: none; padding: 0; font-size: 11.5px; line-height: 1.45; }
ul, ol { margin: 1mm 0 4mm; padding-left: 5.5mm; }
li { margin: 0 0 1.6mm; page-break-inside: avoid; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 4mm; page-break-inside: avoid; break-inside: avoid; }
th {
  text-align: left; font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
  color: #78716C; font-weight: 600; padding: 0 3mm 2mm 0; border-bottom: 1px solid #C9C0B4;
}
td { font-size: 12px; padding: 2mm 3mm 2mm 0; border-bottom: 1px solid #E4DDD2; vertical-align: top; }
td code { white-space: nowrap; }
"""


def render(src: Path) -> str:
    body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    title = src.stem
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>"
        + title
        + "</title><style>"
        + CSS
        + "</style></head><body>"
        + body
        + "</body></html>"
    )


def print_pdf(html: Path, pdf: Path) -> None:
    if pdf.exists():
        pdf.unlink()
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        ],
        check=True,
        timeout=90,
    )


def main() -> None:
    for name, pdf_name, _label in DOCS:
        src = ROOT / name
        html = OUT / f"{src.stem}.html"
        html.write_text(render(src), encoding="utf-8")
        pdf = OUT / pdf_name
        print_pdf(html, pdf)
        print(pdf.name, pdf.stat().st_size)


if __name__ == "__main__":
    main()
