# -*- coding: utf-8 -*-
"""Convierte el guion Markdown a PDF de alta calidad (HTML + Chromium print-to-pdf)."""
import re, os, subprocess, sys
import markdown

BASE = "/home/jhank/Desktop/ModeloCNN"
SRC = os.path.join(BASE, "ENTREGABLES_FINALES", "GUION_EXPOSICION_DIAPOSITIVA_CODIGO_GLOSARIO.md")
OUT_PDF = os.path.join(BASE, "ENTREGABLES_FINALES", "GUION_EXPOSICION_DIAPOSITIVA_CODIGO_GLOSARIO.pdf")
TMP_HTML = "/tmp/guion_exposicion.html"

with open(SRC, "r", encoding="utf-8") as f:
    md = f.read()

# --- Convertir fórmulas $$...$$ a bloques legibles (sin MathJax) ---
REPS = [
    (r"\\frac{N}{K \\cdot n_c}", "N / (K·n_c)"),
    (r"\\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}", "exp(z_i/T) / Σ_j exp(z_j/T)"),
    (r"\\mathcal{L}_{FL}", "L_FL"),
    (r"\\hat{p}_i", "p̂_i"),
    (r"\\qquad", "        "),
    (r"\\lambda", "λ"),
    (r"\\alpha", "α"),
    (r"\\gamma", "γ"),
    (r"\\cdot", "·"),
    (r"\\log", "log"),
    (r"\\exp", "exp"),
    (r"\\sum_j", "Σ_j"),
    (r"\\le", "≤"),
    (r"\\ge", "≥"),
]
def math_text(s):
    for pat, rep in REPS:
        s = re.sub(pat, rep, s)
    return s.strip()

def repl_display(m):
    return f'<div class="formula">{math_text(m.group(1))}</div>'

md = re.sub(r"\$\$(.+?)\$\$", repl_display, md, flags=re.S)

html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

CSS = """
@page { size: A4; margin: 14mm 12mm 15mm 12mm; }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', 'Noto Sans', sans-serif; font-size: 9.4pt; line-height: 1.4;
       color: #17202a; margin: 0; }
h1 { font-size: 17pt; color: #1B5E20; border-bottom: 3px solid #00C853;
     padding-bottom: 5px; margin: 0 0 10px; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 12.5pt; color: #0F172A; margin: 16px 0 6px;
     border-left: 4px solid #00C853; padding-left: 8px; page-break-after: avoid; }
h3 { font-size: 10.6pt; color: #1B5E20; margin: 12px 0 4px; page-break-after: avoid; }
h4 { font-size: 10pt; color: #0F172A; margin: 10px 0 4px; page-break-after: avoid; }
p { margin: 5px 0; }
ul, ol { margin: 4px 0; padding-left: 20px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0;
        font-size: 8.3pt; page-break-inside: auto; }
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
       padding: 1px 4px; border-radius: 3px; font-size: 8.5pt; }
pre { background: #0F172A; color: #E2E8F0; padding: 8px 10px; border-radius: 6px;
      font-size: 7.8pt; overflow-x: auto; page-break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; color: inherit; padding: 0; }
.formula { font-family: 'DejaVu Sans Mono', monospace; background: #eef6ff;
           border: 1px solid #b9d4ea; border-radius: 6px; padding: 8px;
           margin: 8px 0; text-align: center; font-size: 9pt; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #cfd8dc; margin: 12px 0; }
a { color: #0b57a4; text-decoration: none; }
strong { color: #0F172A; }
"""

HTML = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Guión de Exposición — EcoClasificador UCC</title>
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
    print("stderr:", r.stderr.strip()[:400])
print("PDF:", OUT_PDF, os.path.getsize(OUT_PDF) if os.path.exists(OUT_PDF) else "NO GENERADO")
