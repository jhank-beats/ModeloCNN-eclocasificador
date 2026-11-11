# -*- coding: utf-8 -*-
"""
Generador de diapositivas — EcoClasificador Móvil con Visión Artificial y Edge AI
Historia oficial (Guía Maestra): 70.88% -> 92.49% con EfficientNetV2 + Focal Loss + Ensamble
18 diapositivas, 16:9. 2-4 diapositivas por cuaderno.
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = "/home/jhank/Desktop/ModeloCNN"
REC = os.path.join(BASE, "presentacion_recursos", "graficas_cuadernos_2_y_4")
CAP = os.path.join(BASE, "presentacion_recursos", "capturas")
PRE = os.path.join(BASE, "presentacion_recursos")
M12 = os.path.join(BASE, "momento_1_y_2_base")
R90 = os.path.join(M12, "resultados_optimizacion_90")

# ---- Paleta ----
BG      = RGBColor(0x0F, 0x17, 0x2A)
BG2     = RGBColor(0x1E, 0x29, 0x3B)
GREEN   = RGBColor(0x00, 0xC8, 0x53)
GREEN_D = RGBColor(0x1B, 0x5E, 0x20)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
MUTED   = RGBColor(0x94, 0xA3, 0xB8)
LIGHT   = RGBColor(0xE2, 0xE8, 0xF0)
AMBER   = RGBColor(0xF5, 0x9E, 0x0B)
RED     = RGBColor(0xEF, 0x44, 0x44)
BLUE    = RGBColor(0x38, 0xBD, 0xF8)

FONT = "Calibri"
SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]
TOTAL = 18

def set_bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color

def add_rect(slide, x, y, w, h, color, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh

def add_text(slide, x, y, w, h, runs, size=18, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
             space_after=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(runs, str):
        runs = [(runs, {})]
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    p.space_after = Pt(space_after)
    for txt, ov in runs:
        r = p.add_run(); r.text = txt
        f = r.font
        f.name = FONT
        f.size = Pt(ov.get("size", size))
        f.bold = ov.get("bold", bold)
        f.color.rgb = ov.get("color", color)
        f.italic = ov.get("italic", False)
    return tb

def add_bullets(slide, x, y, w, h, items, size=17, gap=8, color=LIGHT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, tuple):
            txt = it[0]; lvl = it[1] if len(it) > 1 else 0
            col = it[2] if len(it) > 2 else color
            b = it[3] if len(it) > 3 else False
            sz = it[4] if len(it) > 4 else size
        else:
            txt = it; lvl = 0; col = color; b = False; sz = size
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(gap)
        p.line_spacing = 1.05
        bullet = "•  " if lvl == 0 else "–  "
        r = p.add_run(); r.text = bullet + txt
        f = r.font; f.name = FONT; f.size = Pt(sz); f.color.rgb = col; f.bold = b
    return tb

def add_image_fit(slide, path, x, y, max_w, max_h, card=True):
    iw, ih = Image.open(path).size
    ar = iw / ih
    box_ar = max_w / max_h
    if ar > box_ar:
        w = max_w; h = int(max_w / ar)
    else:
        h = max_h; w = int(max_h * ar)
    cx = x + (max_w - w) // 2
    cy = y + (max_h - h) // 2
    if card:
        add_rect(slide, x, y, max_w, max_h, WHITE)
    slide.shapes.add_picture(path, cx, cy, w, h)

def add_header(slide, title, subtitle=None, kicker=None):
    add_rect(slide, 0, 0, SW, Inches(0.14), GREEN)
    if kicker:
        add_text(slide, Inches(0.55), Inches(0.28), Inches(12.2), Inches(0.3),
                 kicker.upper(), size=12, color=GREEN, bold=True)
        ty = Inches(0.58)
    else:
        ty = Inches(0.38)
    add_text(slide, Inches(0.55), ty, Inches(12.2), Inches(0.7),
             title, size=28, color=WHITE, bold=True)
    if subtitle:
        add_text(slide, Inches(0.55), ty + Inches(0.58), Inches(12.2), Inches(0.4),
                 subtitle, size=14, color=MUTED)

def add_footer(slide, num):
    add_rect(slide, 0, SH - Inches(0.32), SW, Inches(0.32), BG2)
    add_text(slide, Inches(0.55), SH - Inches(0.30), Inches(9), Inches(0.28),
             "EcoClasificador UCC · Electiva III: Visión Artificial y ML", size=9, color=MUTED)
    add_text(slide, Inches(12.2), SH - Inches(0.30), Inches(0.9), Inches(0.28),
             f"{num} / {TOTAL}", size=9, color=MUTED, align=PP_ALIGN.RIGHT)

def add_table(slide, x, y, w, h, data, col_w=None, fs=13, header_fill=GREEN_D):
    rows = len(data); cols = len(data[0])
    gfx = slide.shapes.add_table(rows, cols, x, y, w, h)
    table = gfx.table
    if col_w:
        for i, cw in enumerate(col_w):
            table.columns[i].width = cw
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = FONT
                    run.font.size = Pt(fs)
                    run.font.color.rgb = WHITE if r == 0 else LIGHT
                    run.font.bold = (r == 0)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else BG2
    return table

def card(slide, x, y, w, h, title, lines, title_color=GREEN):
    add_rect(slide, x, y, w, h, BG2)
    add_rect(slide, x, y, w, Inches(0.06), title_color)
    add_text(slide, x + Inches(0.18), y + Inches(0.16), w - Inches(0.36), Inches(0.4),
             title, size=16, color=title_color, bold=True)
    add_bullets(slide, x + Inches(0.18), y + Inches(0.62), w - Inches(0.36), h - Inches(0.75),
                lines, size=13, gap=6, color=LIGHT)

# =====================================================================
# 1 — PORTADA
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_rect(s, 0, Inches(1.0), SW, Inches(0.06), GREEN)
add_text(s, Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.4),
         "UNIVERSIDAD COOPERATIVA DE COLOMBIA · FACULTAD DE INGENIERÍA", size=14, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.6), Inches(1.9), Inches(12.1), Inches(1.4),
         "EcoClasificador Móvil con Visión Artificial y Edge AI", size=38, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.6), Inches(3.2), Inches(12.1), Inches(0.9),
         "Clasificación Automática de 19 Clases de Residuos Sólidos Urbanos y Peligrosos en Dispositivos de Bajo Consumo",
         size=19, color=LIGHT, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.6), Inches(4.25), Inches(12.1), Inches(1.0),
         [("70.88 %  →  92.49 %", {"size": 46, "color": GREEN, "bold": True})], align=PP_ALIGN.CENTER)
add_text(s, Inches(0.6), Inches(5.25), Inches(12.1), Inches(0.4),
         "de exactitud mediante Optimización Convolucional, Focal Loss y Edge Computing", size=15, color=MUTED, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.6), Inches(6.0), Inches(12.1), Inches(0.4),
         [("Autor: ", {"color": MUTED}), ("Juan Camilo Murillo Moncaleano", {"color": WHITE, "bold": True}),
          ("      Docente: ", {"color": MUTED}), ("Prof. Óscar Díaz Triana", {"color": WHITE, "bold": True})],
         size=16, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.6), Inches(6.45), Inches(12.1), Inches(0.4),
         "Asignatura: Electiva III — Visión Artificial y Machine Learning", size=14, color=MUTED, align=PP_ALIGN.CENTER)

# =====================================================================
# 2 — LA CRISIS DE LOS RESIDUOS
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "La Crisis de los Residuos: el problema está en la fuente", kicker="Bloque 1 · Problemática")
add_text(s, Inches(0.55), Inches(1.6), Inches(7.0), Inches(0.5),
         [("Menos del ", {"color": LIGHT}), ("17 %", {"color": RED, "bold": True, "size": 20}),
          (" de los residuos urbanos se recicla en Colombia y América Latina.", {"color": LIGHT})], size=19)
add_bullets(s, Inches(0.55), Inches(2.3), Inches(7.0), Inches(4.3), [
    (">80 % termina enterrado en rellenos sanitarios a punto de colapsar (Doña Juana, El Carrasco).", 0),
    ("La falla crítica ocurre en la fuente: el ciudadano no sabe cómo categorizar residuos complejos.", 0),
    ("Residuos peligrosos (bombillas con mercurio, aceites, pilas, biosanitarios) se mezclan con reciclables.", 0),
    ("Resultado: toneladas de material aprovechable contaminadas y riesgo para los recicladores de oficio.", 0),
], size=16, gap=12)
add_rect(s, Inches(7.9), Inches(1.7), Inches(4.9), Inches(4.9), BG2)
add_text(s, Inches(8.15), Inches(1.9), Inches(4.4), Inches(0.4),
         "LAS 10 CLASES PELIGROSAS CRÍTICAS", size=14, color=GREEN, bold=True)
add_bullets(s, Inches(8.15), Inches(2.4), Inches(4.4), Inches(4.0), [
    ("Bombillas (vapor de mercurio)", 0), ("Aceites y Químicos", 0),
    ("Pilas y Baterías (Cd, Pb)", 0), ("Medicamentos", 0),
    ("Residuos Sanitarios", 0), ("RAEE (electrónicos)", 0),
    ("Escombros y Cerámica", 0), ("Neumáticos y Gomas", 0),
    ("Madera Procesada", 0), ("Tetra Pak", 0),
], size=14, gap=7, color=LIGHT)
add_footer(s, 2)

# =====================================================================
# 3 — SOLUCIÓN (layout corregido: 3 tarjetas + imagen con espacio propio)
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "La Solución: EcoClasificador UCC", kicker="Bloque 1 · Propuesta de valor")
cw = Inches(3.95); gap = Inches(0.25); x0 = Inches(0.55); cy = Inches(1.65); ch = Inches(2.2)
card(s, x0, cy, cw, ch, "Edge AI (IA en el borde)", [
    ("Inferencia local en el teléfono.", 0),
    ("Modelo TFLite de 6.4 MB.", 0),
    ("Offline, privado, sin Internet.", 0),
])
card(s, x0 + (cw+gap), cy, cw, ch, "19 clases (taxonomía amplia)", [
    ("Supera 3–6 clases de laboratorio.", 0),
    ("Incluye peligrosos y sanitarios.", 0),
    ("Resolución 2184 de 2019.", 0),
])
card(s, x0 + 2*(cw+gap), cy, cw, ch, "Respuesta en tiempo real", [
    ("Menos de 25 ms por imagen.", 0),
    ("Contenedor normativo.", 0),
    ("Blanco, negro, verde o rojo.", 0),
])
add_image_fit(s, os.path.join(PRE, "metodologia_espanol.png"),
              Inches(0.55), Inches(4.15), Inches(12.2), Inches(2.85), card=False)
add_footer(s, 3)

# =====================================================================
# 4 — CUADERNO 1a: RECOLECCIÓN GUIADA (SCRAPING)
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 1 · Recolección Guiada de Imágenes (Web Scraping)", kicker="Bloque 2 · Hilo conductor (1/4)")
add_bullets(s, Inches(0.55), Inches(1.75), Inches(6.6), Inches(4.9), [
    ("Punto de partida: dataset público RealWaste (U. Wollongong) = 9 clases estándar.", 0),
    ("+10 clases críticas creadas para la realidad regional colombiana.", 0, GREEN, True),
    ("Motor Bing Image Crawler (icrawler) con 35+ consultas semánticas por clase, en español e inglés.", 0),
    ("Filtros de búsqueda: type=photo, color=color, min_size 220×220.", 0),
    ("Meta contractual: 1,000 fotografías limpias por clase (10 × 1,000 = 10,000).", 0),
    ("Descarga idempotente: preserva lo ya obtenido y no re-descarga duplicados.", 0),
], size=15, gap=10)
add_rect(s, Inches(7.4), Inches(1.75), Inches(5.4), Inches(4.9), BG2)
add_text(s, Inches(7.65), Inches(1.95), Inches(4.9), Inches(0.4),
         "EJEMPLOS DE CONSULTAS SEMÁNTICAS", size=14, color=GREEN, bold=True)
add_bullets(s, Inches(7.65), Inches(2.45), Inches(4.9), Inches(4.0), [
    ('"used aa aaa batteries pile photo -cartoon -drawing -vector"', 0, MUTED, False, 12),
    ('"bombillas focos fluorescentes fundidos desecho foto real"', 0, MUTED, False, 12),
    ('"escombros construcción ladrillos cerámica azulejos foto real"', 0, MUTED, False, 12),
    ('"mascarillas guantes desechados basura hospitalaria foto real"', 0, MUTED, False, 12),
    ("Los términos negativos (-vector, -diagram) evitan infografías y dibujos.", 1, LIGHT, False, 12),
], size=12, gap=8)
add_footer(s, 4)

# =====================================================================
# 5 — CUADERNO 1b: LIMPIEZA AUTOMÁTICA
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 1 · Limpieza Automática y Filtrado de Calidad", kicker="Bloque 2 · Hilo conductor (2/4)")
add_text(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.4),
         [("Cada imagen descargada pasa por un filtro de calidad estricto antes de entrar al dataset:", {"color": LIGHT})], size=14)
add_bullets(s, Inches(0.55), Inches(2.05), Inches(6.6), Inches(4.4), [
    ("Archivos corruptos y formatos no soportados.", 0),
    ("Resolución baja (<200 px) y aspect ratio extremo.", 0),
    ("Demasiado oscura, quemada o sin contraste.", 0),
    ("Borrosa / plana (varianza Laplaciana < 15).", 0),
    ("Fondo plano (infografía) y pocos colores (dibujo).", 0),
    ("Demasiado texto (Sobel + morfología).", 0),
    ("Duplicados por hash perceptual dHash (Hamming ≤ 2).", 0, GREEN, True),
], size=14, gap=8)
add_rect(s, Inches(7.4), Inches(1.75), Inches(5.4), Inches(4.9), BG2)
add_text(s, Inches(7.65), Inches(1.95), Inches(4.9), Inches(0.4),
         "LIMPIEZA AVANZADA (ANTI-CATÁLOGO)", size=14, color=GREEN, bold=True)
add_bullets(s, Inches(7.65), Inches(2.45), Inches(4.9), Inches(2.0), [
    ("Fondo blanco de estudio/catálogo (>70 % borde blanco).", 0),
    ("Afiches y diagramas con texto estructurado residual.", 0),
    ("Dibujos y vectores por saturación artificial.", 0),
], size=13, gap=7)
add_text(s, Inches(7.65), Inches(4.6), Inches(4.9), Inches(1.9),
         [("Depurador visual (Gradio): ", {"color": GREEN, "bold": True}),
          ("revisión humana con 1 clic para descartar y deshacer, garantizando 0 residuos fuera de dominio.", {"color": LIGHT})],
         size=13)
add_footer(s, 5)

# =====================================================================
# 6 — CUADERNO 1c: NORMALIZACIÓN CON YOLOv8
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 1 · Normalización Inteligente con YOLOv8", kicker="Bloque 2 · Hilo conductor (3/4)")
add_bullets(s, Inches(0.55), Inches(1.75), Inches(6.6), Inches(4.9), [
    ("Problema: el resize directo deforma (squashing) y el letterbox añade bordes artificiales que la CNN aprende.", 0),
    ("YOLOv8n detecta el residuo y devuelve su caja delimitadora [x1,y1,x2,y2] (conf 0.15).", 0, GREEN, True),
    ("Se expande a proporción cuadrada 1:1 con margen de contexto del 8 %.", 0),
    ("Resize a 180×180 px con filtro Lanczos de alta calidad (JPEG q95).", 0),
    ("Respaldo (fallback): recorte central cuadrado para texturas continuas (escombros, vegetación).", 0),
    ("Resultado: el residuo llena el encuadre, sin distorsión geométrica ni franjas.", 0),
], size=14, gap=9)
add_image_fit(s, os.path.join(PRE, "comparativa_yolo_top2.png"),
              Inches(7.2), Inches(1.7), Inches(5.6), Inches(5.0))
add_footer(s, 6)

# =====================================================================
# 7 — CUADERNO 1d: AUDITORÍA MD5 + PARTICIÓN
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 1 · Auditoría de Integridad y Partición Estratificada", kicker="Bloque 2 · Hilo conductor (4/4)")
add_bullets(s, Inches(0.55), Inches(1.75), Inches(6.4), Inches(4.9), [
    ("Auditoría criptográfica (MD5): elimina copias idénticas descargadas de distintos servidores.", 0),
    ("Contradicciones inter-clase: una misma imagen en dos carpetas distintas se elimina automáticamente.", 0),
    ("Partición estratificada y persistente 70 / 15 / 15.", 0, GREEN, True),
    ("7,966 entrenamiento · 1,707 validación · 1,707 prueba estricta.", 1),
    ("Guardada en CSV estático → reproducibilidad científica total.", 0),
], size=15, gap=10)
add_image_fit(s, os.path.join(REC, "cuaderno2_distribucion_clases_dataset.png"),
              Inches(7.0), Inches(1.75), Inches(5.8), Inches(4.9))
add_footer(s, 7)

# =====================================================================
# 8 — CUADERNO 2a: ARQUITECTURA Y TRANSFER LEARNING
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 2 · Modelo Padre: MobileNetV2 (Transfer Learning)", kicker="Bloque 2 · Baseline (1/2)")
add_bullets(s, Inches(0.55), Inches(1.7), Inches(7.0), Inches(2.2), [
    ("Backbone MobileNetV2 pre-entrenado en ImageNet-1k (aprendizaje por transferencia).", 0),
    ("Cabeza: GAP → Dense(256, ReLU) → Dropout(0.3) → Dense(19, Softmax).", 0),
], size=15, gap=8)
add_rect(s, Inches(0.55), Inches(3.6), Inches(7.0), Inches(1.35), BG2)
add_text(s, Inches(0.75), Inches(3.72), Inches(6.6), Inches(1.1),
         [("Input(180,180,3) → MobileNetV2 (congelado) → GAP → Dense(256, ReLU) → Dropout(0.3) → Dense(19, Softmax)",
           {"size": 14, "color": GREEN, "bold": True})], align=PP_ALIGN.CENTER, line_spacing=1.1)
add_bullets(s, Inches(0.55), Inches(5.1), Inches(7.0), Inches(1.7), [
    ("Fase 1 (warmup): backbone congelado, solo la cabeza, Adam lr=1e-3.", 0),
    ("Fase 2 (fine-tuning): últimas 30 capas convolucionales, lr=1e-4.", 0),
], size=14, gap=7)
card(s, Inches(7.9), Inches(1.75), Inches(4.9), Inches(4.9), "¿POR QUÉ MOBILENETV2?", [
    ("Convoluciones depthwise separables: 8–9× menos operaciones.", 0),
    ("Residuales invertidos con cuellos lineales.", 0),
    ("Ligero: ideal para móvil (base del Edge AI).", 0),
])
add_footer(s, 8)

# =====================================================================
# 9 — CUADERNO 2b: RESULTADOS Y SOBREAJUSTE
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 2 · Resultados: el 'Techo de Cristal' (70.88 %)", kicker="Bloque 2 · Baseline (2/2)")
add_bullets(s, Inches(0.55), Inches(1.7), Inches(6.4), Inches(2.6), [
    ("98 % de exactitud en entrenamiento…", 0),
    ("…pero solo 70.88 % en validación y test.", 0, RED, True),
    ("Diagnóstico: sobreajuste severo (overfitting).", 0),
    ("La red memoriza fondos e iluminación en vez de texturas del residuo.", 0),
], size=15, gap=10)
add_rect(s, Inches(0.55), Inches(4.5), Inches(6.4), Inches(1.9), BG2)
add_text(s, Inches(0.75), Inches(4.68), Inches(6.0), Inches(1.6),
         [("Mitigación inicial: ", {"color": GREEN, "bold": True}),
          ("data augmentation dinámico (rotación ±15°, brillo, volteo horizontal) para no memorizar fondos.", {"color": LIGHT})],
         size=14)
add_image_fit(s, os.path.join(REC, "cuaderno2_matriz_confusion_baseline_mobilenetv2.png"),
              Inches(7.0), Inches(1.7), Inches(5.8), Inches(4.9))
add_footer(s, 9)

# =====================================================================
# 10 — CUADERNO 3a: PLANTEAMIENTO Y VARIANTES
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 3 · Defensa Técnica de Regularización (Ablación)", kicker="Bloque 2 · Regularización (1/2)")
add_text(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.4),
         [("Pregunta de investigación: ", {"color": LIGHT}),
          ("¿basta con regularizadores clásicos para romper el sobreajuste?", {"color": GREEN, "bold": True})], size=15)
card(s, Inches(0.55), Inches(2.1), Inches(3.95), Inches(2.7), "Variante A · Dropout puro", [
    ("Desconecta el 30 % de las neuronas en cada pasada.", 0),
    ("Evita co-adaptación entre neuronas.", 0),
], title_color=BLUE)
card(s, Inches(4.7), Inches(2.1), Inches(3.95), Inches(2.7), "Variante B · + L2 (Ridge)", [
    ("Penaliza pesos grandes: λ·Σw².", 0),
    ("Contrae suavemente la magnitud de los pesos.", 0),
], title_color=AMBER)
card(s, Inches(8.85), Inches(2.1), Inches(3.95), Inches(2.7), "Variante C · + L1 (Lasso)", [
    ("Fuerza dispersión (sparsity): λ·Σ|w|.", 0),
    ("Pone en cero conexiones espurias de fondo.", 0),
], title_color=GREEN)
add_text(s, Inches(0.55), Inches(5.0), Inches(12.2), Inches(1.4),
         [("Condición experimental: ", {"color": MUTED, "bold": True}),
          ("mismas semillas, mismo dataset y mismo backbone MobileNetV2 → comparación justa y reproducible.",
           {"color": LIGHT})], size=14)
add_footer(s, 10)

# =====================================================================
# 11 — CUADERNO 3b: RESULTADOS DE ABLACIÓN + CONCLUSIÓN
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 3 · Resultados de la Ablación y Conclusión", kicker="Bloque 2 · Regularización (2/2)")
add_table(s, Inches(0.55), Inches(1.7), Inches(6.3), Inches(2.2), [
    ["Variante", "Brecha (val-train)", "Exactitud test"],
    ["A · Dropout puro", "1.1263", "66.56 %"],
    ["B · Dropout + L2", "1.1015", "67.19 %"],
    ["C · Dropout + L1", "0.9969", "68.54 %"],
], col_w=[Inches(2.6), Inches(2.3), Inches(1.4)], fs=14)
add_rect(s, Inches(0.55), Inches(4.2), Inches(6.3), Inches(2.3), BG2)
add_text(s, Inches(0.75), Inches(4.38), Inches(5.9), Inches(2.0),
         [("Conclusión científica: ", {"color": GREEN, "bold": True}),
          ("la regularización reduce la brecha de memorización, pero la exactitud sigue < 70 %. El cuello de botella está en la capacidad del backbone, la función de pérdida y la aumentación.", {"color": LIGHT})],
         size=14)
add_image_fit(s, os.path.join(M12, "curvas_regularizacion_defensa.png"),
              Inches(7.0), Inches(1.7), Inches(5.8), Inches(4.9))
add_footer(s, 11)

# =====================================================================
# 12 — CUADERNO 4a: EFFICIENTNETV2-B0
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 4 · Nueva Arquitectura: EfficientNetV2-B0", kicker="Bloque 2 · Optimización avanzada (1/4)")
add_bullets(s, Inches(0.55), Inches(1.75), Inches(6.6), Inches(4.9), [
    ("Sustituye el backbone por EfficientNetV2-B0 (bloques Fused-MBConv).", 0, GREEN, True),
    ("Atención Squeeze-and-Excitation: mayor sensibilidad a texturas locales.", 1),
    ("ImageNet Top-1: 78.7 % (frente al 72.0 % de MobileNetV2).", 0),
    ("include_preprocessing=True: recibe píxeles nativos [0,255].", 0),
    ("Sobre esta base se aplica un protocolo de 7 innovaciones.", 0),
], size=15, gap=10)
add_image_fit(s, os.path.join(REC, "cuaderno4_curvas_entrenamiento_loss_acc_lr.png"),
              Inches(7.2), Inches(1.75), Inches(5.6), Inches(4.9))
add_footer(s, 12)

# =====================================================================
# 13 — CUADERNO 4b: MIXUP / CUTMIX / FOCAL LOSS
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 4 · MixUp, CutMix y Focal Loss", kicker="Bloque 2 · Optimización avanzada (2/4)")
add_bullets(s, Inches(0.55), Inches(1.7), Inches(6.4), Inches(2.5), [
    ("MixUp (α=0.2): mezcla convexa de imágenes y etiquetas; límites de decisión suaves.", 0),
    ("CutMix (α=1.0): pega un parche de una imagen sobre otra; evita depender de un solo detalle.", 0),
    ("Focal Loss (γ=2.0, α=0.25): redirige el gradiente a clases difíciles y minoritarias.", 0),
], size=14, gap=9)
add_rect(s, Inches(0.55), Inches(4.25), Inches(6.4), Inches(2.2), BG2)
add_text(s, Inches(0.75), Inches(4.4), Inches(6.0), Inches(0.8),
         [("L_FL(p_t) = −α (1 − p_t)^γ · log(p_t)", {"size": 22, "color": GREEN, "bold": True})], align=PP_ALIGN.CENTER)
add_text(s, Inches(0.75), Inches(5.25), Inches(6.0), Inches(1.0),
         [("Si p_t = 0.95 → (1−0.95)² = 0.0025 → suprime el gradiente un 99.75 % y se enfoca en pilas, aceites, sanitarios.",
           {"size": 13, "color": MUTED})], align=PP_ALIGN.CENTER)
add_image_fit(s, os.path.join(REC, "mixup_cutmix_visualizacion.png"),
              Inches(7.0), Inches(1.7), Inches(5.8), Inches(4.9))
add_footer(s, 13)

# =====================================================================
# 14 — CUADERNO 4c: CONFIDENT LEARNING + ENSAMBLE
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Cuaderno 4 · Poda de Ruido Web y Ensamble Calibrado", kicker="Bloque 2 · Optimización avanzada (3/4)")
add_bullets(s, Inches(0.55), Inches(1.7), Inches(6.4), Inches(4.9), [
    ("Confident Learning (Cleanlab/MIT): poda de muestras con p_etiqueta < 0.12.", 0, GREEN, True),
    ("133 muestras fuera de dominio depuradas (7.8 % del test).", 1),
    ("Ensamble homogéneo: P = 0.80·Focal + 0.20·Base.", 0, GREEN, True),
    ("Multi-Angle TTA: 0.50·orig + 0.30·flipH + 0.20·flipV.", 0),
    ("Temperature Scaling (T=0.75): agudiza la probabilidad correcta.", 0),
], size=14, gap=9)
add_image_fit(s, os.path.join(REC, "cuaderno4_galeria_muestras_ruido_web_detectadas.png"),
              Inches(7.0), Inches(1.7), Inches(5.8), Inches(4.9))
add_footer(s, 14)

# =====================================================================
# 15 — CUADERNO 4d: RESULTADOS Y MATRIZ DE CONFUSIÓN ⭐
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Resultados · Matriz de Confusión y Métricas Finales", kicker="Bloque 3 · Métricas")
def _chips(slide):
    n = 4; gap = Inches(0.18); cw = (Inches(7.0) - gap*(n-1)) // n
    for i,(lab,val) in enumerate([("Top-1","92.49 %"),("Top-2","98.24 %"),("Top-3","99.80 %"),("F1 Macro","0.921")]):
        cx = Inches(0.55) + i*(cw+gap)
        add_rect(slide, cx, Inches(1.6), cw, Inches(0.95), BG2)
        add_text(slide, cx+Inches(0.05), Inches(1.68), cw-Inches(0.1), Inches(0.35), val, size=19, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, cx+Inches(0.05), Inches(2.08), cw-Inches(0.1), Inches(0.35), lab, size=11, color=MUTED, align=PP_ALIGN.CENTER)
_chips(s)
add_text(s, Inches(0.55), Inches(2.75), Inches(7.0), Inches(0.4), "Evolución de la exactitud (Top-1):", size=14, color=MUTED)
add_bullets(s, Inches(0.55), Inches(3.15), Inches(7.0), Inches(3.2), [
    ("Cuaderno 2 · MobileNetV2 baseline → 70.88 %", 0),
    ("Cuaderno 3 · + regularización L1 → 68.54 %", 0),
    ("Cuaderno 4 (Parte I) · EfficientNetV2 + MixUp/CutMix → 82.66 %", 0),
    ("Cuaderno 4 (Ensamble crudo) → 85.10 %", 0),
    ("Cuaderno 4 (Final curado) · Ensamble + Poda → 92.49 %", 0, GREEN, True),
], size=13, gap=7)
add_image_fit(s, os.path.join(REC, "cuaderno4_matriz_confusion_final_ensamble_92acc.png"),
              Inches(7.7), Inches(1.6), Inches(5.1), Inches(5.3))
add_footer(s, 15)

# =====================================================================
# 16 — DESPLIEGUE EDGE AI · TFLITE
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Despliegue Edge AI · TensorFlow Lite", kicker="Bloque 4 · Producción móvil")
add_bullets(s, Inches(0.55), Inches(1.75), Inches(6.4), Inches(4.9), [
    ("Cuantización INT8 (rango dinámico): 113 MB → 6.4 MB.", 0, GREEN, True),
    ("Latencia de inferencia: 15–25 ms en CPU móvil comercial.", 0),
    ("App Android: EcoClasificador_Bienal_UCC.apk (100 % offline).", 0),
    ("Pre-recorte con YOLOv8-Nano + estandarización a 180×180.", 0),
    ("Consumo energético mínimo: jornadas continuas sin recalentar.", 0),
], size=16, gap=12)
add_image_fit(s, os.path.join(BASE, "modelos", "flujo_entrenamiento_y_conversion_modelo.png"),
              Inches(7.0), Inches(1.75), Inches(5.8), Inches(4.9))
add_footer(s, 16)

# =====================================================================
# 17 — DEMOSTRACIÓN PLATAFORMA WEB
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Demostración · Plataforma Web (Ensamble Calibrado)", kicker="Bloque 4 · Inferencia en vivo")
add_text(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.4),
         [("Inferencia en vivo con residuos peligrosos y de manejo especial — ", {"color": LIGHT}),
          ("Bombillas, Escombros y Residuos Sanitarios", {"color": GREEN, "bold": True})], size=15)
imgs = [("web_bombillas.png", "Bombillas — 97.0 %"),
        ("web_escombros.png", "Escombros — 98.2 %"),
        ("web_residuos_sanitarios.png", "Residuos Sanitarios — 99.4 %")]
x0 = Inches(0.55); cw = Inches(3.9); gap = Inches(0.35)
for i, (fn, cap) in enumerate(imgs):
    x = x0 + i * (cw + gap)
    add_rect(s, x, Inches(2.05), cw, Inches(4.55), BG2)
    add_image_fit(s, os.path.join(CAP, fn), x + Inches(0.05), Inches(2.12), cw - Inches(0.1), Inches(3.95), card=False)
    add_text(s, x, Inches(6.15), cw, Inches(0.4), cap, size=14, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
add_footer(s, 17)

# =====================================================================
# 18 — CONCLUSIONES E IMPACTO (ODS)
# =====================================================================
s = prs.slides.add_slide(BLANK); set_bg(s)
add_header(s, "Conclusiones e Impacto Regional", kicker="Bloque 5 · Cierre")
add_bullets(s, Inches(0.55), Inches(1.75), Inches(7.0), Inches(4.9), [
    ("Llevamos el modelo del cuaderno al campo: 70.88 % → 92.49 % (Top-2 98.24 %).", 0, GREEN, True),
    ("Ciclo completo CRISP-ML(Q): ingesta → limpieza → YOLO → entrenamiento → Edge AI.", 0),
    ("Edge AI viable: 6.4 MB, 20 ms, 100 % offline en teléfonos de gama media.", 0),
    ("Metodología reproducible y defendible (ablasión, poda de ruido, calibración).", 0),
], size=16, gap=12)
add_rect(s, Inches(7.9), Inches(1.75), Inches(4.9), Inches(4.9), BG2)
add_text(s, Inches(8.15), Inches(1.95), Inches(4.4), Inches(0.4),
         "APORTE A LA AGENDA 2030 (ODS)", size=14, color=GREEN, bold=True)
add_bullets(s, Inches(8.15), Inches(2.5), Inches(4.4), Inches(3.9), [
    ("ODS 11 — Ciudades y comunidades sostenibles (Meta 11.6).", 0),
    ("ODS 12 — Producción y consumo responsables (Meta 12.5).", 0),
    ("ODS 3 — Salud y bienestar (Meta 3.9).", 0),
    ("ODS 13 — Acción por el clima.", 0),
], size=14, gap=9)
add_footer(s, 18)

OUT = os.path.join(BASE, "ENTREGABLES_FINALES", "Exposicion_Final_EcoClasificador_92acc.pptx")
prs.save(OUT)
print("Guardado:", OUT)
print("Diapositivas:", len(prs.slides._sldIdLst))
