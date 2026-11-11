# -*- coding: utf-8 -*-
"""Convierte glosario.md a PDF (HTML + Chromium print-to-pdf)."""
import os, subprocess
import markdown

BASE = "/home/jhank/Desktop/ModeloCNN"
SRC = os.path.join(BASE, "glosario.md")
OUT_PDF = os.path.join(BASE, "glosario_ml.pdf")
TMP_HTML = "/tmp/glosario_ml.html"

with open(SRC, "r", encoding="utf-8") as f:
    md = f.read()

html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

CSS = """
@page { size: A4; margin: 15mm 13mm 16mm 13mm; }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', 'Noto Sans', sans-serif; font-size: 9.6pt; line-height: 1.42;
       color: #17202a; margin: 0; }
h1 { font-size: 17pt; color: #1B5E20; border-bottom: 3px solid #00C853;
     padding-bottom: 5px; margin: 0 0 4px; }
h2 { font-size: 11pt; color: #4a5568; margin: 2px 0 14px; font-weight: normal; }
h3 { font-size: 12.5pt; color: #0F172A; margin: 16px 0 6px;
     border-left: 4px solid #00C853; padding-left: 8px; page-break-after: avoid; }
h4 { font-size: 10.6pt; color: #1B5E20; margin: 12px 0 4px; page-break-after: avoid; }
p { margin: 5px 0; }
ul, ol { margin: 4px 0; padding-left: 20px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0;
        font-size: 8.5pt; page-break-inside: auto; }
th { background: #1B5E20; color: #fff; text-align: left; padding: 4px 6px;
     border: 1px solid #1B5E20; }
td { border: 1px solid #cfd8dc; padding: 3px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #f5f8f5; }
tr { page-break-inside: avoid; }
blockquote { border-left: 4px solid #00C853; background: #f0fdf4;
             margin: 8px 0; padding: 6px 10px; color: #14532d; page-break-inside: avoid; }
blockquote p { margin: 3px 0; }
blockquote code { background: #dcfce7; }
code { font-family: 'DejaVu Sans Mono', monospace; background: #eceff1;
       padding: 1px 4px; border-radius: 3px; font-size: 8.6pt; }
pre { background: #0F172A; color: #E2E8F0; padding: 8px 10px; border-radius: 6px;
      font-size: 8.2pt; overflow-x: auto; page-break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; color: inherit; padding: 0; }
hr { border: none; border-top: 1px solid #cfd8dc; margin: 12px 0; }
strong { color: #0F172A; }
"""

HTML = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Glosario Técnico — ML y Deep Learning</title>
<style>{CSS}</style></head>
<body>{html_body}</body></html>"""

with open(TMP_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

cmd = [
    "chromium", "--headless", "--disable-gpu", "--no-sandbox",
    "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
    f"file://{TMP_HTML}"
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
print("chromium rc:", r.returncode)
if r.stderr.strip():
    print("stderr:", r.stderr.strip()[:300])
print("PDF:", OUT_PDF, os.path.getsize(OUT_PDF) if os.path.exists(OUT_PDF) else "NO GENERADO")
